# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from ultralytics.utils.metrics import OKS_SIGMA, RLE_WEIGHT
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, bbox_siou, bbox_mpdiou, bbox_inner_iou, bbox_piou, probiou
from .tal import bbox2dist, rbox2dist


# ============================================================================
# NWD (Normalized Wasserstein Distance) Loss
# ============================================================================
# Reference: Wang et al., "A Normalized Gaussian Wasserstein Distance for
# Tiny Object Detection" (2021) — https://arxiv.org/abs/2110.13389
#
# Key idea: model bboxes as 2D Gaussian distributions and measure their
# distance via Wasserstein metric. For small objects, NWD is more stable
# than IoU because small positional errors don't cause the metric to
# collapse to zero.

def wasserstein_distance(mu1, sigma1, mu2, sigma2):
    """Compute the squared 2-Wasserstein distance between two 2D Gaussians."""
    # W₂² = ||mu1 - mu2||² + Tr(sigma1 + sigma2 - 2*(sigma1^(1/2)*sigma2*sigma1^(1/2))^(1/2))
    # For axis-aligned bboxes, sigma is diagonal → simplifies
    mean_diff_sq = (mu1 - mu2).pow(2).sum(dim=-1)  # [N, M]
    sigma_diff_sq = (sigma1.sqrt() - sigma2.sqrt()).pow(2).sum(dim=-1)  # [N, M]
    return mean_diff_sq + sigma_diff_sq


def bbox_nwd(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    constant: float = 12.8,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute Normalized Wasserstein Distance (NWD) between predicted and target bboxes.

    Each bbox (cx, cy, w, h) is modeled as a 2D Gaussian:
        mu = (cx, cy)
        sigma = diag(w²/4, h²/4)  (simplified from Gaussian ellipse)

    Args:
        pred_boxes:   [N, 4] predicted bboxes in xyxy coordinates
        target_boxes: [N, 4] target bboxes in xyxy coordinates
        constant:     normalization constant (default 12.8, as in original paper)
        eps:          small constant for numerical stability

    Returns:
        nwd: [N] NWD similarity in [0, 1] (1 = perfect match, 0 = completely different)
    """
    if pred_boxes.shape[-1] != 4 or target_boxes.shape[-1] != 4:
        raise ValueError("bbox_nwd expects xyxy boxes with four coordinates")
    wh1 = pred_boxes[..., 2:4] - pred_boxes[..., 0:2]
    cx1 = (pred_boxes[..., 0:2] + pred_boxes[..., 2:4]) / 2.0
    wh2 = target_boxes[..., 2:4] - target_boxes[..., 0:2]
    cx2 = (target_boxes[..., 0:2] + target_boxes[..., 2:4]) / 2.0

    # Gaussian parameters: mu = center, sigma proportional to wh
    mu1 = cx1
    sigma1 = (wh1 / 2.0).clamp(min=eps) ** 2  # w²/4, h²/4 — clamp for stability
    mu2 = cx2
    sigma2 = (wh2 / 2.0).clamp(min=eps) ** 2

    w2_sq = wasserstein_distance(mu1, sigma1, mu2, sigma2)
    nwd = torch.exp(-torch.sqrt(w2_sq + eps) / constant)
    return nwd


def bbox_gcd(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute aligned Gaussian Combined Distance (GCD) box similarity.

    This is a PyTorch/YOLO adaptation of the official implementation from
    https://github.com/MArKkwanGuan/mmdet-GCD. GCD symmetrically normalizes
    centre displacement and width/height error by both boxes, making the
    similarity scale invariant. Inputs are aligned ``xyxy`` boxes and the
    returned similarity lies in ``(0, 1]``.
    """
    if pred_boxes.shape[-1] != 4 or target_boxes.shape[-1] != 4:
        raise ValueError("bbox_gcd expects xyxy boxes with four coordinates")

    pred_center = (pred_boxes[..., :2] + pred_boxes[..., 2:]) / 2.0
    target_center = (target_boxes[..., :2] + target_boxes[..., 2:]) / 2.0
    center_delta = pred_center - target_center

    pred_wh = (pred_boxes[..., 2:] - pred_boxes[..., :2]).clamp(min=eps)
    target_wh = (target_boxes[..., 2:] - target_boxes[..., :2]).clamp(min=eps)
    wh_delta = pred_wh - target_wh

    center_by_pred = (center_delta / pred_wh).pow(2).sum(dim=-1)
    shape_by_target = (wh_delta / target_wh).pow(2).sum(dim=-1) / 4.0
    center_by_target = (center_delta / target_wh).pow(2).sum(dim=-1)
    shape_by_pred = (wh_delta / pred_wh).pow(2).sum(dim=-1) / 4.0

    gcd_sq = (center_by_pred + shape_by_target + center_by_target + shape_by_pred) / 2.0
    # The official exp(-sqrt(gcd_sq)) has an infinite derivative at the exact
    # optimum. Subtracting sqrt(eps) after stabilising the root keeps
    # similarity=1 for identical boxes while retaining finite gradients.
    distance = torch.sqrt(gcd_sq.clamp(min=0.0) + eps) - math.sqrt(eps)
    return torch.exp(-distance)


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth.

        Returns per-element loss (b, a, nc) — caller handles .sum() / target_scores_sum.
        """
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight
            )
        return loss


class QualityFocalLoss(nn.Module):
    """Quality Focal Loss (GFLv1) — combines classification with IoU quality score.

    QFL = -|y - sigma|^beta * ((1-y)log(1-sigma) + y*log(sigma))
    where y is the IoU score (continuous [0,1]), not a discrete label.

    Reference: Li et al., "Generalized Focal Loss" (NeurIPS 2020)
    https://arxiv.org/abs/2006.04388
    """

    def __init__(self, beta: float = 2.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute QFL between predictions and continuous IoU-based labels.

        Returns per-element loss (b, a, nc) — caller handles .sum() / target_scores_sum.
        """
        pred_sigmoid = pred.sigmoid()
        scale_factor = (pred_sigmoid - label).abs().pow(self.beta)
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        return loss * scale_factor


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on
    hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25, reduction: str = "mean"):
        """Initialize FocalLoss class with focusing and balancing parameters.

        Args:
            gamma (float): Focusing parameter; higher values down-weight easy examples more.
            alpha (float): Class-balancing factor; reduces loss contribution from majority class.
            reduction (str): 'mean' for scalar output (legacy default), 'none' for per-element loss
                compatible with v8DetectionLoss normalization (sum / target_scores_sum).
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance.

        When reduction='none', returns per-element focal loss with no aggregation,
        allowing v8DetectionLoss to normalize via .sum() / target_scores_sum.
        """
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        if self.reduction == "none":
            return loss
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes.

    Supports GCD, mixed IoU + NWD, WIoU and Focal-WIoU box losses.
    Controlled via environment variables:
        YOLO_USE_GCD=1         replace the IoU box term with pure GCD
        YOLO_USE_NWD=1         enable NWD
        YOLO_NWD_RATIO=0.5     NWD mix ratio
        YOLO_NWD_CONSTANT=12.8 NWD normalization constant
        YOLO_USE_WIOU=1        enable WIoU (wise-IoU v1)
        YOLO_USE_FOCAL_WIOU=1  enable Focal-WIoU (overrides WIoU if both set)
    """

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

        # GCD (Gaussian Combined Distance). DFL remains a separate term.
        self.use_gcd = os.getenv("YOLO_USE_GCD", "0") == "1"

        # NWD
        self.use_nwd = os.getenv("YOLO_USE_NWD", "0") == "1"
        self.nwd_ratio = float(os.getenv("YOLO_NWD_RATIO", "0.5"))
        self.nwd_constant = float(os.getenv("YOLO_NWD_CONSTANT", "12.8"))

        # WIoU / Focal-WIoU
        self.use_wiou = os.getenv("YOLO_USE_WIOU", "0") == "1"
        self.use_focal_wiou = os.getenv("YOLO_USE_FOCAL_WIOU", "0") == "1"

        if self.use_gcd and (self.use_nwd or self.use_wiou or self.use_focal_wiou):
            raise ValueError("GCD is a complete box-term replacement and cannot be combined with NWD/WIoU flags")

        # IoU variant (ciou/diou/siou/mpdiou/inner_iou/piou)
        self.iou_type = os.getenv("YOLO_IOU_TYPE", "ciou")
        self.inner_iou_ratio = float(os.getenv("YOLO_INNER_IOU_RATIO", "0.7"))

    def _wiou_v3_loss(self, pred_boxes, target_boxes, delta=1.0, alpha=1.9, eps=1e-7):
        """Wise-IoU v3: dynamic non-monotonic focusing mechanism.

        v1:  L = R_WIoU * L_IoU,  R_WIoU = exp(dist²/(W²+H²))
        v3:  L = r * R_WIoU * L_IoU

        where r = β / (δ * α^(β-δ))  — dynamic gradient gain
              β = L_IoU* / moving_mean(L_IoU)  — outlier degree (* = detached)
              δ=1.0, α=1.9  (paper defaults)

        v3 gives highest gradient gain to "ordinary quality" anchors and
        suppresses extreme outliers, providing more stable training than v1.
        """
        # ── R_WIoU: distance-attention (same as v1) ──
        pred_cx = (pred_boxes[..., 0] + pred_boxes[..., 2]) / 2.0
        pred_cy = (pred_boxes[..., 1] + pred_boxes[..., 3]) / 2.0
        target_cx = (target_boxes[..., 0] + target_boxes[..., 2]) / 2.0
        target_cy = (target_boxes[..., 1] + target_boxes[..., 3]) / 2.0

        x1 = torch.min(pred_boxes[..., 0], target_boxes[..., 0])
        y1 = torch.min(pred_boxes[..., 1], target_boxes[..., 1])
        x2 = torch.max(pred_boxes[..., 2], target_boxes[..., 2])
        y2 = torch.max(pred_boxes[..., 3], target_boxes[..., 3])
        W = (x2 - x1).clamp(min=eps).detach()
        H = (y2 - y1).clamp(min=eps).detach()
        dist_sq = (pred_cx - target_cx).pow(2) + (pred_cy - target_cy).pow(2)
        # Keep every per-box term as (N, 1).  bbox_iou() returns (N, 1),
        # while indexing xyxy coordinates above produces (N,).  Multiplying
        # those shapes directly broadcasts to (N, N), which previously made
        # the WIoU box loss thousands of times too large in YOLOv8-FPDW.
        r_wiou = torch.exp(dist_sq / (W.pow(2) + H.pow(2) + eps)).unsqueeze(-1)

        iou = bbox_iou(pred_boxes, target_boxes, xywh=False, CIoU=False)
        l_iou = 1.0 - iou  # per-element IoU loss

        # ── v3: dynamic non-monotonic focusing ──
        # β = outlier degree (ratio of individual loss to mean, detached)
        with torch.no_grad():
            l_iou_mean = l_iou.mean().clamp(min=eps)
            beta = l_iou / l_iou_mean  # >1 = worse than average (outlier)
            # r = β / (δ * α^(β-δ))
            r = beta / (delta * torch.pow(alpha, beta - delta) + eps)
            r = r.detach()

        return r * r_wiou * l_iou  # per-box loss (n,)

    def _focal_wiou_loss(self, pred_boxes, target_boxes, gamma=0.5, eps=1e-7):
        """Focal-WIoU: WIoU v1 + focal weighting (IoU^γ) for hard-example mining."""
        pred_cx = (pred_boxes[..., 0] + pred_boxes[..., 2]) / 2.0
        pred_cy = (pred_boxes[..., 1] + pred_boxes[..., 3]) / 2.0
        target_cx = (target_boxes[..., 0] + target_boxes[..., 2]) / 2.0
        target_cy = (target_boxes[..., 1] + target_boxes[..., 3]) / 2.0

        x1 = torch.min(pred_boxes[..., 0], target_boxes[..., 0])
        y1 = torch.min(pred_boxes[..., 1], target_boxes[..., 1])
        x2 = torch.max(pred_boxes[..., 2], target_boxes[..., 2])
        y2 = torch.max(pred_boxes[..., 3], target_boxes[..., 3])
        W = (x2 - x1).clamp(min=eps).detach()
        H = (y2 - y1).clamp(min=eps).detach()

        dist_sq = (pred_cx - target_cx).pow(2) + (pred_cy - target_cy).pow(2)
        r_wiou = torch.exp(dist_sq / (W.pow(2) + H.pow(2) + eps)).unsqueeze(-1)

        iou = bbox_iou(pred_boxes, target_boxes, xywh=False, CIoU=False)
        focal_weight = iou.pow(gamma).detach()

        return focal_weight * r_wiou * (1.0 - iou)  # per-box loss (n,)

    def forward(
        self,
        pred_dist, pred_bboxes, anchor_points, target_bboxes,
        target_scores, target_scores_sum, fg_mask, imgsz, stride,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        if self.use_focal_wiou:
            loss_iou_val = self._focal_wiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            loss_iou = (loss_iou_val * weight).sum() / target_scores_sum
        elif self.use_wiou:
            loss_iou_val = self._wiou_v3_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            loss_iou = (loss_iou_val * weight).sum() / target_scores_sum
        elif self.use_gcd:
            gcd = bbox_gcd(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            loss_iou = ((1.0 - gcd) * weight.squeeze(-1)).sum() / target_scores_sum
        else:
            # Select IoU variant
            if self.iou_type == "eiou":
                pred = pred_bboxes[fg_mask]
                target = target_bboxes[fg_mask]
                iou = bbox_iou(pred, target, xywh=False, CIoU=False)
                pw = (pred[..., 2] - pred[..., 0]).clamp(min=1e-7)
                ph = (pred[..., 3] - pred[..., 1]).clamp(min=1e-7)
                tw = (target[..., 2] - target[..., 0]).clamp(min=1e-7)
                th = (target[..., 3] - target[..., 1]).clamp(min=1e-7)
                pcx = (pred[..., 0] + pred[..., 2]) * 0.5
                pcy = (pred[..., 1] + pred[..., 3]) * 0.5
                tcx = (target[..., 0] + target[..., 2]) * 0.5
                tcy = (target[..., 1] + target[..., 3]) * 0.5
                cw = (torch.maximum(pred[..., 2], target[..., 2]) - torch.minimum(pred[..., 0], target[..., 0])).clamp(min=1e-7)
                ch = (torch.maximum(pred[..., 3], target[..., 3]) - torch.minimum(pred[..., 1], target[..., 1])).clamp(min=1e-7)
                center_term = ((pcx - tcx).pow(2) + (pcy - tcy).pow(2)) / (cw.pow(2) + ch.pow(2) + 1e-7)
                shape_term = (pw - tw).pow(2) / (cw.pow(2) + 1e-7) + (ph - th).pow(2) / (ch.pow(2) + 1e-7)
                loss_eiou = 1.0 - iou + (center_term + shape_term).unsqueeze(-1)
                loss_iou = (loss_eiou * weight).sum() / target_scores_sum
                iou = None
            elif self.iou_type == "siou":
                iou = bbox_siou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
            elif self.iou_type == "diou":
                iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, DIoU=True)
            elif self.iou_type == "mpdiou":
                iou = bbox_mpdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
            elif self.iou_type == "inner_iou":
                iou = bbox_inner_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, ratio=self.inner_iou_ratio)
            elif self.iou_type == "piou":
                iou = bbox_piou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
            else:  # ciou (default)
                iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            if iou is not None:
                loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.use_nwd:
            # pred_bboxes/target_bboxes are expressed in feature-grid units at
            # this point. NWD's normalization constant is defined in input-image
            # pixels, so restore the per-anchor stride before measuring W2.
            stride_map = stride.unsqueeze(0) if stride.ndim == 2 else stride
            stride_fg = stride_map.expand(pred_bboxes.shape[0], -1, -1)[fg_mask]
            pred_boxes_px = pred_bboxes[fg_mask] * stride_fg
            target_boxes_px = target_bboxes[fg_mask] * stride_fg
            nwd = bbox_nwd(pred_boxes_px, target_boxes_px, constant=self.nwd_constant)
            loss_nwd = ((1.0 - nwd) * weight.squeeze(-1)).sum() / target_scores_sum
            loss_iou = (1.0 - self.nwd_ratio) * loss_iou + self.nwd_ratio * loss_nwd

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]; target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist_f = pred_dist * stride
            pred_dist_f[..., 0::2] /= imgsz[1]; pred_dist_f[..., 1::2] /= imgsz[0]
            loss_dfl = (F.l1_loss(pred_dist_f[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight).sum() / target_scores_sum

        return loss_iou, loss_dfl


class RLELoss(nn.Module):
    """Residual Log-Likelihood Estimation Loss.

    Args:
        use_target_weight (bool): Option to use weighted loss.
        size_average (bool): Option to average the loss by the batch_size.
        residual (bool): Option to add L1 loss and let the flow learn the residual error distribution.

    References:
        https://arxiv.org/abs/2107.11291
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/regression_loss.py
    """

    def __init__(self, use_target_weight: bool = True, size_average: bool = True, residual: bool = True):
        """Initialize RLELoss with target weight and residual options.

        Args:
            use_target_weight (bool): Whether to use target weights for loss calculation.
            size_average (bool): Whether to average the loss over elements.
            residual (bool): Whether to include residual log-likelihood term.
        """
        super().__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual

    def forward(
        self, sigma: torch.Tensor, log_phi: torch.Tensor, error: torch.Tensor, target_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            sigma (torch.Tensor): Output sigma, shape (N, D).
            log_phi (torch.Tensor): Output log_phi, shape (N).
            error (torch.Tensor): Error, shape (N, D).
            target_weight (torch.Tensor): Weights across different joint types, shape (N).
        """
        log_sigma = torch.log(sigma)
        loss = log_sigma - log_phi.unsqueeze(1)

        if self.residual:
            loss += torch.log(sigma * 2) + torch.abs(error)

        if self.use_target_weight:
            assert target_weight is not None, "'target_weight' should not be None when 'use_target_weight' is True."
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum()


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = rbox2dist(
                target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5], reg_max=self.dfl_loss.reg_max - 1
            )
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = rbox2dist(target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5])
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class MultiChannelDiceLoss(nn.Module):
    """Criterion class for computing multi-channel Dice losses."""

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        """Initialize MultiChannelDiceLoss with smoothing and reduction options.

        Args:
            smooth (float): Smoothing factor to avoid division by zero.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate multi-channel Dice loss between predictions and targets."""
        assert pred.size() == target.size(), "the size of predict and target must be equal."

        pred = pred.sigmoid()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        dice_loss = dice_loss.mean(dim=1)

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class BCEDiceLoss(nn.Module):
    """Criterion class for computing combined BCE and Dice losses."""

    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        """Initialize BCEDiceLoss with BCE and Dice weight factors.

        Args:
            weight_bce (float): Weight factor for BCE loss component.
            weight_dice (float): Weight factor for Dice loss component.
        """
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = MultiChannelDiceLoss(smooth=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined BCE and Dice loss between predictions and targets."""
        _, _, mask_h, mask_w = pred.shape
        if tuple(target.shape[-2:]) != (mask_h, mask_w):  # downsample to the same size as pred
            target = F.interpolate(target, (mask_h, mask_w), mode="nearest")
        return self.weight_bce * self.bce(pred, target) + self.weight_dice * self.dice(pred, target)


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        # Classification loss: BCE (default) / Focal / Varifocal / QualityFocal
        self.use_focal_loss = getattr(h, "use_focal_loss", False)
        self.use_varifocal = os.getenv("YOLO_USE_VARIFOCAL", "0") == "1"
        self.use_qfl = os.getenv("YOLO_USE_QFL", "0") == "1"
        self.ordinal = os.getenv("YOLO_ORDINAL", "0") == "1"
        self.ordinal_tau = float(os.getenv("YOLO_ORDINAL_TAU", "0.5"))

        # OQL — Ordinal Quality Loss for ordered-class ripeness detection
        self.use_oql = os.getenv("YOLO_USE_OQL", "0") == "1"
        self.oql_lambda = float(os.getenv("YOLO_OQL_LAMBDA", "0.2"))
        self.oql_expect_weight = float(os.getenv("YOLO_OQL_EXPECT_WEIGHT", "1.0"))
        self.oql_kl_weight = float(os.getenv("YOLO_OQL_KL_WEIGHT", "0.0"))
        self.oql_neighbor = float(os.getenv("YOLO_OQL_NEIGHBOR", "0.1"))
        self.oql_tau = float(os.getenv("YOLO_OQL_TAU", "0.5"))

        # QCL — Quality Consistency Loss (aligns classification confidence with IoU)
        self.use_qcl = os.getenv("YOLO_USE_QCL", "0") == "1"
        self.qcl_lambda = float(os.getenv("YOLO_QCL_LAMBDA", "0.1"))
        self.qcl_type = os.getenv("YOLO_QCL_TYPE", "bce")
        self.qcl_detach_iou = os.getenv("YOLO_QCL_DETACH_IOU", "1") == "1"
        self.qcl_min_iou = float(os.getenv("YOLO_QCL_MIN_IOU", "0.0"))

        # Quality head loss — for QualityDetect head (supervises quality branch with IoU)
        self.use_quality_head_loss = os.getenv("YOLO_USE_QHEAD_LOSS", "1") == "1"
        self.qhead_lambda = float(os.getenv("YOLO_QHEAD_LAMBDA", "0.2"))
        self.pf_prompt_lambda = float(os.getenv("YOLO_PF_PROMPT_LAMBDA", "0.1"))
        self.pf_centerness_lambda = float(os.getenv("YOLO_PF_CENTERNESS_LAMBDA", "0.2"))

        # Maturity head loss — for MaturityDetect head
        self.maturity_loss_type = os.getenv("YOLO_MATURITY_LOSS", "gaussian_nll")
        self.maturity_lambda = float(model.yaml.get("maturity_lambda", os.getenv("YOLO_MATURITY_LAMBDA", "0.1")))
        self.maturity_variance_min = float(model.yaml.get("maturity_variance_min", 1e-4))
        self.maturity_variance_max = float(model.yaml.get("maturity_variance_max", 1e2))
        if not 0 < self.maturity_variance_min < self.maturity_variance_max:
            raise ValueError(
                "Maturity variance bounds must satisfy 0 < min < max, got "
                f"{self.maturity_variance_min}, {self.maturity_variance_max}."
            )
        maturity_targets = model.yaml.get("maturity_targets")
        if maturity_targets is None:
            maturity_targets = list(range(self.nc))
        if not isinstance(maturity_targets, (list, tuple)) or len(maturity_targets) != self.nc:
            raise ValueError(
                f"maturity_targets must contain one numeric value for each of the {self.nc} classes, "
                f"but got {maturity_targets!r}."
            )
        try:
            self.maturity_targets = torch.tensor(maturity_targets, device=device, dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"maturity_targets must contain numeric values, but got {maturity_targets!r}.") from exc

        maturity_ignore_classes = model.yaml.get("maturity_ignore_classes", [])
        if not isinstance(maturity_ignore_classes, (list, tuple)):
            raise ValueError("maturity_ignore_classes must be a list of class indices.")
        self.maturity_ignore_classes = tuple(sorted({int(class_id) for class_id in maturity_ignore_classes}))
        if any(class_id < 0 or class_id >= self.nc for class_id in self.maturity_ignore_classes):
            raise ValueError(
                f"maturity_ignore_classes contains an index outside [0, {self.nc - 1}]: "
                f"{self.maturity_ignore_classes}."
            )

        # Log classification loss type only once (v8DetectionLoss instantiated 2×: train+val)
        if self.use_varifocal:
            vf_gamma = float(os.getenv("YOLO_VF_GAMMA", "2.0"))
            vf_alpha = float(os.getenv("YOLO_VF_ALPHA", "0.75"))
            self.bce = VarifocalLoss(gamma=vf_gamma, alpha=vf_alpha)
        elif self.use_qfl:
            qfl_beta = float(os.getenv("YOLO_QFL_BETA", "2.0"))
            self.bce = QualityFocalLoss(beta=qfl_beta)
        elif self.use_focal_loss:
            focal_gamma = getattr(h, "focal_gamma", 1.5)
            focal_alpha = getattr(h, "focal_alpha", 0.25)
            self.bce = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, reduction="none")
        else:
            self.bce = nn.BCEWithLogitsLoss(reduction="none")

        tal_topk = int(os.getenv("YOLO_TAL_TOPK", str(tal_topk)))
        tal_topk2_env = os.getenv("YOLO_TAL_TOPK2", "")
        tal_topk2 = int(tal_topk2_env) if tal_topk2_env else tal_topk2
        tal_alpha = float(os.getenv("YOLO_TAL_ALPHA", "0.5"))
        tal_beta = float(os.getenv("YOLO_TAL_BETA", "6.0"))

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=tal_alpha,
            beta=tal_beta,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def _apply_ordinal_targets(self, target_scores: torch.Tensor) -> torch.Tensor:
        """Convert one-hot targets to maturity-distance-aware ordinal soft labels."""
        if not self.ordinal or self.nc <= 1:
            return target_scores

        quality = target_scores.sum(-1, keepdim=True)
        pos_mask = quality.squeeze(-1) > 0
        if not pos_mask.any():
            return target_scores

        cls_idx = target_scores.argmax(-1)
        maturity = self.maturity_targets.to(device=target_scores.device, dtype=target_scores.dtype)
        target_maturity = maturity[cls_idx]
        distances = (maturity.view(1, 1, -1) - target_maturity.unsqueeze(-1)).abs()
        weights = torch.exp(-distances / max(self.ordinal_tau, 1e-6))
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        ordinal_scores = weights * quality
        return torch.where(pos_mask.unsqueeze(-1), ordinal_scores, target_scores)

    def _ordinal_quality_loss(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
        target_scores_sum: torch.Tensor,
    ) -> torch.Tensor:
        """Ordinal Quality Loss — expected-stage regression + optional ordinal KL.

        Designed for ordered-class ripeness (unripe=0, semi-ripe=1, ripe=2).
        Penalises distant misclassifications more than adjacent ones by regressing
        the expected class index against the ground-truth stage.  Target scores
        from the task-aligned assigner serve as per-sample quality weights so that
        high-IoU positives contribute more.

        Args:
            pred_scores:    [B, A, nc]  raw classification logits
            target_scores:  [B, A, nc]  TAL assigner target scores (soft targets)
            fg_mask:        [B, A]      boolean foreground mask
            target_scores_sum: scalar    normalisation denominator

        Returns:
            scalar loss (0.0 when OQL is disabled or no foreground samples exist)
        """
        if not self.use_oql or self.nc <= 1 or fg_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_scores.device, dtype=pred_scores.dtype)

        # --- positive samples ---
        pred_pos = pred_scores[fg_mask]          # [N, nc]
        target_pos = target_scores[fg_mask]      # [N, nc]
        target_cls = target_pos.argmax(-1)        # [N]
        quality = target_pos.sum(-1).detach().clamp(min=0)  # [N]

        # --- expected stage regression ---
        p = pred_pos.softmax(dim=-1)                        # [N, nc]
        stages = self.maturity_targets.to(device=pred_scores.device, dtype=p.dtype)  # [nc]
        pred_stage = (p * stages).sum(-1)                    # [N] expected maturity
        target_stage = stages[target_cls]                    # [N]

        expect = F.smooth_l1_loss(pred_stage, target_stage, reduction="none")  # [N]
        expect = (expect * quality).sum() / target_scores_sum.clamp_min(1.0)

        # --- optional ordinal KL term ---
        kl = torch.tensor(0.0, device=pred_scores.device, dtype=pred_scores.dtype)
        if self.oql_kl_weight > 0:
            target_maturity = stages[target_cls].unsqueeze(-1)                      # [N, 1]
            distances = (stages.view(1, -1) - target_maturity).abs()                # [N, nc]
            ordinal_dist = torch.exp(-distances / max(self.oql_tau, 1e-6))           # [N, nc]
            ordinal_dist = ordinal_dist / ordinal_dist.sum(-1, keepdim=True).clamp_min(1e-9)

            kl = F.kl_div(
                F.log_softmax(pred_pos, dim=-1), ordinal_dist, reduction="none"
            ).sum(-1)                                                    # [N]
            kl = (kl * quality).sum() / target_scores_sum.clamp_min(1.0)

        return self.oql_expect_weight * expect + self.oql_kl_weight * kl

    def _quality_consistency_loss(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
        target_scores_sum: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Quality Consistency Loss — aligns per-class confidence with localisation IoU.

        Penalises the gap between the sigmoid confidence of the assigned class and
        the predicted-vs-target box IoU so that "high score" implies "well-localised".

        Args:
            pred_scores:    [B, A, nc]  classification logits
            pred_bboxes:    [B, A, 4]   decoded predicted boxes (stride units)
            target_bboxes:  [B, A, 4]   assigned GT boxes (image pixel units)
            target_scores:  [B, A, nc]  TAL assigner target scores
            fg_mask:        [B, A]      boolean foreground mask
            target_scores_sum: scalar    normalisation denominator
            stride_tensor:  [A, 1]      per-anchor stride

        Returns:
            scalar loss (0.0 when QCL is disabled or no foreground samples exist)
        """
        if not self.use_qcl or fg_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_scores.device, dtype=pred_scores.dtype)

        # Convert predicted boxes to image pixel coordinates
        pred_boxes_img = pred_bboxes * stride_tensor  # [B, A, 4]

        # --- foreground samples ---
        pred_pos = pred_scores[fg_mask]          # [N, nc]
        pred_box_pos = pred_boxes_img[fg_mask]    # [N, 4]
        target_box_pos = target_bboxes[fg_mask]   # [N, 4]
        target_score_pos = target_scores[fg_mask]  # [N, nc]

        # --- class with highest target score ---
        target_cls = target_score_pos.argmax(-1)                        # [N]

        # --- predicted logit for the assigned class ---
        cls_logit = pred_pos.gather(1, target_cls.unsqueeze(-1)).squeeze(-1)  # [N] raw logit

        # --- localisation quality = IoU between predicted box and GT ---
        quality = bbox_iou(pred_box_pos, target_box_pos, xywh=False, CIoU=False)       # [N] or [N,1]
        quality = quality.squeeze(-1).clamp(min=self.qcl_min_iou, max=1.0)
        if self.qcl_detach_iou:
            quality = quality.detach()

        # --- per-sample weight from TAL target scores ---
        sample_weight = target_score_pos.sum(-1).detach().clamp(min=0)  # [N]

        # --- quality-vs-confidence loss ---
        if self.qcl_type == "bce":
            # BCEWithLogits = sigmoid + BCE, AMP-safe
            q_loss = F.binary_cross_entropy_with_logits(cls_logit, quality, reduction="none")
        elif self.qcl_type == "l1":
            q_loss = (cls_logit.sigmoid().float() - quality).abs()
        elif self.qcl_type == "smoothl1":
            q_loss = F.smooth_l1_loss(cls_logit.sigmoid().float(), quality, reduction="none")
        else:
            raise ValueError(f"Unknown qcl_type: {self.qcl_type}")

        return (q_loss * sample_weight).sum() / target_scores_sum.clamp_min(1.0)

    def _quality_head_loss(
        self,
        pred_quality: torch.Tensor | None,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
        target_scores_sum: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Supervise the QualityDetect quality branch with per-anchor IoU.

        Args:
            pred_quality:   [B, A, 1]  quality logits from QualityDetect (or None)
            pred_bboxes:    [B, A, 4]  decoded predicted boxes (stride units)
            target_bboxes:  [B, A, 4]  assigned GT boxes (image pixel units)
            target_scores:  [B, A, nc] TAL assigner target scores
            fg_mask:        [B, A]     boolean foreground mask
            target_scores_sum: scalar   normalisation denominator
            stride_tensor:  [A, 1]     per-anchor stride

        Returns:
            scalar loss (0.0 when quality head is absent or no foreground)
        """
        if pred_quality is None or fg_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_bboxes.device, dtype=pred_bboxes.dtype)

        pred_boxes_img = pred_bboxes * stride_tensor          # [B, A, 4]
        pred_q_pos = pred_quality[fg_mask].squeeze(-1)         # [N]
        quality_target = (
            bbox_iou(pred_boxes_img[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=False)
            .detach().squeeze(-1).clamp(0, 1)                               # [N]
        )
        weight = target_scores[fg_mask].sum(-1).detach()       # [N]
        q_loss = F.binary_cross_entropy_with_logits(pred_q_pos, quality_target, reduction="none")
        return (q_loss * weight).sum() / target_scores_sum.clamp_min(1.0)

    def _centerness_head_loss(
        self,
        pred_centerness: torch.Tensor | None,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
        target_scores_sum: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Supervise PF-CGQS centerness at positive anchor locations."""
        if pred_centerness is None or fg_mask.sum() == 0:
            return torch.zeros((), device=target_bboxes.device, dtype=target_bboxes.dtype)
        centers = anchor_points * stride_tensor
        centers = centers.unsqueeze(0).expand(target_bboxes.shape[0], -1, -1)
        left_top = centers - target_bboxes[..., :2]
        right_bottom = target_bboxes[..., 2:] - centers
        distances = torch.cat((left_top, right_bottom), dim=-1)[fg_mask].clamp_min(0)
        left, top, right, bottom = distances.unbind(-1)
        horizontal = torch.minimum(left, right) / torch.maximum(left, right).clamp_min(1e-6)
        vertical = torch.minimum(top, bottom) / torch.maximum(top, bottom).clamp_min(1e-6)
        target = (horizontal * vertical).clamp_min(0).sqrt().detach()
        logits = pred_centerness[fg_mask].squeeze(-1)
        weight = target_scores[fg_mask].sum(-1).detach()
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (loss * weight).sum() / target_scores_sum.clamp_min(1.0)

    def _maturity_head_loss(
        self,
        pred_mu: torch.Tensor | None,
        pred_lv: torch.Tensor | None,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
        target_scores_sum: torch.Tensor,
    ) -> torch.Tensor:
        """Supervise MaturityDetect mean/log-var branches with configured continuous maturity targets.

        The default target for each class is its class index, preserving the
        original three-stage behavior. Model YAML may override this mapping
        when class IDs are not in chronological maturity order.

        Args:
            pred_mu:      [B, A, 1] maturity mean logits (or None)
            pred_lv:      [B, A, 1] maturity log-variance (or None)
            target_scores: [B, A, nc] TAL assigner target scores
            fg_mask:       [B, A] boolean foreground mask
            target_scores_sum: scalar normalisation denominator

        Returns:
            scalar loss
        """
        if pred_mu is None or fg_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_mu.device if pred_mu is not None else
                                target_scores.device, dtype=target_scores.dtype)

        mu_pos = pred_mu[fg_mask].squeeze(-1)  # [N]
        target_cls = target_scores[fg_mask].argmax(-1).long()  # [N] class indices
        weight = target_scores[fg_mask].sum(-1).detach()  # [N]

        # Keep the legacy default (class index as maturity value) while allowing
        # datasets whose label IDs are not in chronological maturity order.
        valid_mask = torch.ones_like(target_cls, dtype=torch.bool)
        if self.maturity_ignore_classes:
            ignored = torch.tensor(self.maturity_ignore_classes, device=target_cls.device)
            valid_mask = ~torch.isin(target_cls, ignored)
        if not valid_mask.any():
            return torch.zeros((), device=pred_mu.device, dtype=pred_mu.dtype)

        mu_pos = mu_pos[valid_mask]
        target_mu = self.maturity_targets[target_cls[valid_mask]].to(mu_pos.dtype)
        weight = weight[valid_mask]

        if self.maturity_loss_type == "smoothl1":
            loss = F.smooth_l1_loss(mu_pos, target_mu, reduction="none")
        elif self.maturity_loss_type == "gaussian_nll":
            # Gaussian NLL is deliberately evaluated in FP32. In FP16,
            # squared error divided by a very small variance can overflow
            # 65,504 even when all model parameters remain finite.
            mu_fp32 = mu_pos.float()
            target_fp32 = target_mu.float()
            lv_fp32 = pred_lv[fg_mask].squeeze(-1)[valid_mask].float()
            var = F.softplus(lv_fp32).clamp(
                min=self.maturity_variance_min,
                max=self.maturity_variance_max,
            )
            loss = 0.5 * ((mu_fp32 - target_fp32).square() / var + var.log())
            if not torch.isfinite(loss).all():
                raise FloatingPointError(
                    "Non-finite MaturityDetect Gaussian NLL after FP32 conversion and variance clamping."
                )
            weight = weight.float()
        else:
            raise ValueError(f"Unknown maturity_loss_type: {self.maturity_loss_type}")

        normalizer = target_scores_sum if not self.maturity_ignore_classes else weight.sum().clamp_min(1.0)
        return (loss * weight).sum() / normalizer.clamp_min(1.0)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and
        target indices.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        # QualityDetect quality branch (optional — None for standard Detect)
        pred_quality = preds.get("quality", None)
        if pred_quality is not None:
            pred_quality = pred_quality.permute(0, 2, 1).contiguous()  # [B, A, 1]
        pred_prompt = preds.get("prompt_scores", None)
        if pred_prompt is not None:
            pred_prompt = pred_prompt.permute(0, 2, 1).contiguous()  # [B, A, 1]
        pred_centerness = preds.get("centerness", None)
        if pred_centerness is not None:
            pred_centerness = pred_centerness.permute(0, 2, 1).contiguous()  # [B, A, 1]
        # MaturityDetect branches (optional)
        pred_mu = preds.get("maturity_mean", None)
        if pred_mu is not None:
            pred_mu = pred_mu.permute(0, 2, 1).contiguous()  # [B, A, 1]
        pred_lv = preds.get("maturity_lv", None)
        if pred_lv is not None:
            pred_lv = pred_lv.permute(0, 2, 1).contiguous()  # [B, A, 1]
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = target_scores.sum().clamp_min(1.0)
        cls_target_scores = self._apply_ordinal_targets(target_scores).to(dtype)

        # Cls loss
        if self.use_varifocal:
            hard_labels = (target_scores > 0).to(dtype)
            loss[1] = self.bce(pred_scores, cls_target_scores, hard_labels).sum() / target_scores_sum
        else:
            loss[1] = self.bce(pred_scores, cls_target_scores).sum() / target_scores_sum  # BCE/QFL/Focal

        # OQL — auxiliary ordinal-quality loss (does not replace BCE, only adds order signal)
        if self.use_oql:
            loss_oql = self._ordinal_quality_loss(pred_scores, target_scores, fg_mask, target_scores_sum)
            loss[1] = loss[1] + self.oql_lambda * loss_oql

        # QCL — quality consistency loss (aligns cls confidence with IoU)
        if self.use_qcl:
            qcl = self._quality_consistency_loss(
                pred_scores, pred_bboxes, target_bboxes, target_scores,
                fg_mask, target_scores_sum, stride_tensor,
            )
            loss[1] = loss[1] + self.qcl_lambda * qcl

        # Quality head loss — supervises QualityDetect quality branch with IoU
        if pred_quality is not None and self.use_quality_head_loss:
            qhead = self._quality_head_loss(
                pred_quality, pred_bboxes, target_bboxes, target_scores,
                fg_mask, target_scores_sum, stride_tensor,
            )
            loss[1] = loss[1] + self.qhead_lambda * qhead

        # CSP local adaptation: supervise the first-pass fruitness mask after
        # merging all maturity classes into one foreground target.
        if pred_prompt is not None:
            fruit_target = target_scores.sum(-1, keepdim=True).clamp(0, 1).to(pred_prompt.dtype)
            prompt_loss = F.binary_cross_entropy_with_logits(pred_prompt, fruit_target, reduction="none")
            loss[1] = loss[1] + self.pf_prompt_lambda * prompt_loss.sum() / target_scores_sum

        if pred_centerness is not None:
            center_loss = self._centerness_head_loss(
                pred_centerness,
                target_bboxes,
                target_scores,
                fg_mask,
                target_scores_sum,
                anchor_points,
                stride_tensor,
            )
            loss[1] = loss[1] + self.pf_centerness_lambda * center_loss

        # Maturity head loss — supervises MaturityDetect mean/log-var branches
        if pred_mu is not None:
            m_loss = self._maturity_head_loss(
                pred_mu, pred_lv, target_scores, fg_mask, target_scores_sum,
            )
            loss[1] = loss[1] + self.maturity_lambda * m_loss

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # loss(box, cls, dfl)

    def parse_output(
        self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Parse model predictions to extract features."""
        return preds[1] if isinstance(preds, tuple) else preds

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """A wrapper for get_assigned_targets_and_loss and parse_output."""
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss * batch_size, loss_detach


class ProposalMaturityLoss(v8DetectionLoss):
    """Loss for class-agnostic fruit proposals plus ROI maturity classification."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk, tal_topk2)
        self.head = model.model[-1]
        self.maturity_nc = self.head.nc
        self.nc = 1
        self.no = 1 + self.reg_max * 4
        self.roi_lambda = float(os.getenv("YOLO_PF_ROI_LAMBDA", "1.0"))
        self.roi_max_samples = int(os.getenv("YOLO_PF_ROI_MAX_SAMPLES", "512"))
        self.assigner = TaskAlignedAssigner(
            topk=int(os.getenv("YOLO_TAL_TOPK", str(tal_topk))),
            num_classes=1,
            alpha=float(os.getenv("YOLO_TAL_ALPHA", "0.5")),
            beta=float(os.getenv("YOLO_TAL_BETA", "6.0")),
            stride=self.stride.tolist(),
            topk2=int(os.getenv("YOLO_TAL_TOPK2")) if os.getenv("YOLO_TAL_TOPK2", "") else tal_topk2,
        )

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        loss = torch.zeros(3, device=self.device)  # box, fruitness+ROI cls, dfl
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_objectness = preds["fruitness"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_objectness.shape[0]
        dtype = pred_objectness.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        object_labels = torch.zeros_like(gt_labels)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_objectness.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            object_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = target_scores.sum().clamp_min(1.0)
        loss[1] = F.binary_cross_entropy_with_logits(pred_objectness, target_scores.to(dtype), reduction="sum")
        loss[1] = loss[1] / target_scores_sum

        if fg_mask.any():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

            # Use assigned predicted proposals, detached only in their geometry,
            # to train the ROI classifier. Gradients still flow into ROI features.
            pred_boxes_img = pred_bboxes * stride_tensor
            boxes_by_image, labels_by_image = [], []
            for batch_index in range(batch_size):
                positive = fg_mask[batch_index].nonzero(as_tuple=False).squeeze(1)
                if positive.numel() > self.roi_max_samples:
                    quality = target_scores[batch_index, positive, 0]
                    positive = positive[quality.topk(self.roi_max_samples).indices]
                boxes = pred_boxes_img[batch_index, positive].detach().clone()
                boxes[:, 0::2].clamp_(0, float(imgsz[1]))
                boxes[:, 1::2].clamp_(0, float(imgsz[0]))
                boxes[:, 2] = torch.maximum(boxes[:, 2], boxes[:, 0] + 1.0)
                boxes[:, 3] = torch.maximum(boxes[:, 3], boxes[:, 1] + 1.0)
                labels = gt_labels[batch_index, target_gt_idx[batch_index, positive], 0].long()
                boxes_by_image.append(boxes)
                labels_by_image.append(labels)
            roi_logits = self.head.classify_proposals(preds["feats"], boxes_by_image)
            roi_labels = torch.cat(labels_by_image, dim=0)
            if roi_logits.numel():
                roi_loss = F.cross_entropy(roi_logits, roi_labels)
                loss[1] = loss[1] + self.roi_lambda * roi_loss

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss * batch_size, loss.detach()


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model, tal_topk, tal_topk2)
        self.overlap = model.args.overlap_mask
        self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        pred_masks, proto = preds["mask_coefficient"].permute(0, 2, 1).contiguous(), preds["proto"]
        loss = torch.zeros(5, device=self.device)  # box, seg, cls, dfl
        if isinstance(proto, tuple) and len(proto) == 2:
            proto, pred_semseg = proto
        else:
            pred_semseg = None
        (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]

        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        if fg_mask.sum():
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                # masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
                proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)

            imgsz = (
                torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_masks.dtype) * self.stride[0]
            )
            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                masks,
                target_gt_idx,
                target_bboxes,
                batch["batch_idx"].view(-1, 1),
                proto,
                pred_masks,
                imgsz,
            )
            if pred_semseg is not None:
                sem_masks = batch["sem_masks"].to(self.device)  # NxHxW
                sem_masks = F.one_hot(sem_masks.long(), num_classes=self.nc).permute(0, 3, 1, 2).float()  # NxCxHxW

                if self.overlap:
                    mask_zero = masks == 0  # NxHxW
                    sem_masks[mask_zero.unsqueeze(1).expand_as(sem_masks)] = 0
                else:
                    batch_idx = batch["batch_idx"].view(-1)  # [total_instances]
                    for i in range(batch_size):
                        instance_mask_i = masks[batch_idx == i]  # [num_instances_i, H, W]
                        if len(instance_mask_i) == 0:
                            continue
                        sem_masks[i, :, instance_mask_i.sum(dim=0) == 0] = 0

                loss[4] = self.bcedice_loss(pred_semseg, sem_masks)
                loss[4] *= self.hyp.box  # seg gain

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
            if pred_semseg is not None:
                loss[4] += (pred_semseg * 0).sum()

        loss[1] *= self.hyp.box  # seg gain
        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if self.overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int = 10):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model, tal_topk, tal_topk2)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        # Pboxes
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        # Bbox loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain

        return loss * batch_size, loss.detach()  # loss(box, pose, kobj, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def _select_target_keypoints(
        self,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        target_gt_idx: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Select target keypoints for each anchor based on batch index and target ground truth index.

        Args:
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).

        Returns:
            (torch.Tensor): Selected keypoints tensor, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        return selected_keypoints

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        # Select target keypoints using helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class PoseLoss26(v8PoseLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation with RLE loss support."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize PoseLoss26 with model parameters and keypoint-specific loss functions including RLE loss."""
        super().__init__(model, tal_topk, tal_topk2)
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        self.rle_loss = None
        self.flow_model = model.model[-1].flow_model if hasattr(model.model[-1], "flow_model") else None
        if self.flow_model is not None:
            self.rle_loss = RLELoss(use_target_weight=True).to(self.device)
            self.target_weights = (
                torch.from_numpy(RLE_WEIGHT).to(self.device) if is_pose else torch.ones(nkpt, device=self.device)
            )

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(6 if self.rle_loss else 5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)  # (b, h*w, 17, 3)

        if self.rle_loss and preds.get("kpts_sigma", None) is not None:
            pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
            pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)  # (b, h*w, 17, 2)
            pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)  # (b, h*w, 17, 5)

        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)

        # Bbox loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            keypoints_loss = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
            loss[1] = keypoints_loss[0]
            loss[2] = keypoints_loss[1]
            if self.rle_loss is not None:
                loss[5] = keypoints_loss[2]

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        if self.rle_loss is not None:
            loss[5] *= self.hyp.rle  # rle gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl, kpt_location, kpt_visibility)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y

    def calculate_rle_loss(self, pred_kpt: torch.Tensor, gt_kpt: torch.Tensor, kpt_mask: torch.Tensor) -> torch.Tensor:
        """Calculate the RLE (Residual Log-likelihood Estimation) loss for keypoints.

        Args:
            pred_kpt (torch.Tensor): Predicted keypoints with sigma, shape (N, kpts_dim) where kpts_dim >= 4.
            gt_kpt (torch.Tensor): Ground truth keypoints, shape (N, kpts_dim).
            kpt_mask (torch.Tensor): Mask for valid keypoints, shape (N, num_keypoints).

        Returns:
            (torch.Tensor): The RLE loss.
        """
        pred_kpt_visible = pred_kpt[kpt_mask]
        gt_kpt_visible = gt_kpt[kpt_mask]
        pred_coords = pred_kpt_visible[:, 0:2]
        pred_sigma = pred_kpt_visible[:, -2:]
        gt_coords = gt_kpt_visible[:, 0:2]

        target_weights = self.target_weights.unsqueeze(0).repeat(kpt_mask.shape[0], 1)
        target_weights = target_weights[kpt_mask]

        pred_sigma = pred_sigma.sigmoid()
        error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)

        # Filter out NaN and Inf values to prevent MultivariateNormal validation errors
        valid_mask = ~(torch.isnan(error) | torch.isinf(error)).any(dim=-1)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_kpt.device)

        error = error[valid_mask]
        error = error.clamp(-100, 100)  # Prevent numerical instability
        pred_sigma = pred_sigma[valid_mask]
        target_weights = target_weights[valid_mask]

        log_phi = self.flow_model.log_prob(error)

        return self.rle_loss(pred_sigma, log_phi, error, target_weights)

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
            rle_loss (torch.Tensor): The RLE loss.
        """
        # Select target keypoints using inherited helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0
        rle_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
                rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask)
            if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss, rle_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model, tal_topk=tal_topk)
        self.assigner = RotatedTaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, angle
        pred_distri, pred_scores, pred_angle = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["angle"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width

        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo26n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = target_scores.sum().clamp_min(1.0)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            weight = target_scores.sum(-1)[fg_mask]
            loss[3] = self.calculate_angle_loss(
                pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
            )  # angle loss
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.angle  # angle gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl, angle)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

    def calculate_angle_loss(self, pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum, lambda_val=3):
        """Calculate oriented angle loss.

        Args:
            pred_bboxes: [N, 5] (x, y, w, h, theta).
            target_bboxes: [N, 5] (x, y, w, h, theta).
            fg_mask: Foreground mask indicating valid predictions.
            weight: Loss weights for each prediction.
            target_scores_sum: Sum of target scores for normalization.
            lambda_val: control the sensitivity to aspect ratio.
        """
        w_gt = target_bboxes[..., 2]
        h_gt = target_bboxes[..., 3]
        pred_theta = pred_bboxes[..., 4]
        target_theta = target_bboxes[..., 4]

        log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))
        scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))

        delta_theta = pred_theta - target_theta
        delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
        ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2

        ang_loss = scale_weight[fg_mask] * ang_loss
        ang_loss = ang_loss * weight

        return ang_loss.sum() / target_scores_sum


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class E2ELoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model, loss_fn=v8DetectionLoss):
        """Initialize E2ELoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
        self.updates = 0
        self.total = 1.0
        # init gain
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        # final gain
        self.final_o2m = 0.1

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1]

    def update(self) -> None:
        """Update the weights for one-to-many and one-to-one losses based on the decay schedule."""
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x) -> float:
        """Calculate the decayed weight for one-to-many loss based on the current update step."""
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model, tal_topk, tal_topk2)
        # NOTE: store following info as it's changeable in __call__
        self.hyp = self.vp_criterion.hyp
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def parse_output(self, preds) -> dict[str, torch.Tensor]:
        """Parse model predictions to extract features."""
        return self.vp_criterion.parse_output(preds)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, preds: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        scores = preds["scores"]
        vnc = scores.shape[1]

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc
        return scores


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model, tal_topk=10):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model, tal_topk)
        self.hyp = self.vp_criterion.hyp

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]
