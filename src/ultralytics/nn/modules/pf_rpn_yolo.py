# Ultralytics AGPL-3.0 License
"""PF-RPN-inspired heads adapted to the FSDH-YOLO dense detection interface.

These modules reproduce the *purpose* of PF-RPN's SIA, CSP and CG-QS mechanisms
without pretending that Grounding-DINO queries can be copied directly into a
YOLO head.  They are intentionally mutually exclusive experiment variants.
"""

from __future__ import annotations

import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv
from .head import Detect, QualityDetect

__all__ = (
    "PFScaleAwareDetect",
    "PFSelfPromptDetect",
    "PFCGQSDetect",
    "FruitProposalMaturityDetect",
)


class _SparseScaleAdapter(nn.Module):
    """Image-conditioned sparse weighting of P3/P4/P5 features.

    PF-RPN SIA updates an object embedding from multi-level visual features.  In
    YOLO there is no text/query stream, so the local adaptation uses one learned
    fruit query to score feature levels and applies a residual, top-k scale gate.
    """

    def __init__(self, channels: tuple[int, ...], hidden: int = 64, topk: int = 2, strength: float = 0.5):
        super().__init__()
        self.nl = len(channels)
        self.topk = max(1, min(int(topk), self.nl))
        self.strength = float(strength)
        self.projections = nn.ModuleList(
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, hidden), nn.SiLU()) for c in channels
        )
        self.fruit_query = nn.Parameter(torch.randn(hidden) / math.sqrt(hidden))
        self.scale_bias = nn.Parameter(torch.zeros(self.nl))

    def forward(self, features: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]:
        descriptors = torch.stack([projection(feature) for projection, feature in zip(self.projections, features)], 1)
        logits = (descriptors * self.fruit_query.view(1, 1, -1)).sum(-1) / math.sqrt(descriptors.shape[-1])
        logits = logits + self.scale_bias
        weights = logits.softmax(dim=1)
        if self.topk < self.nl:
            keep = weights.topk(self.topk, dim=1).indices
            mask = torch.zeros_like(weights).scatter_(1, keep, 1.0)
            weights = weights * mask
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
        # Mean gain remains one; the selector redistributes emphasis instead of
        # globally increasing activation magnitude.
        gains = 1.0 + self.strength * (self.nl * weights - 1.0)
        adapted = [feature * gains[:, index, None, None, None] for index, feature in enumerate(features)]
        return adapted, weights


class PFScaleAwareDetect(Detect):
    """FSDH Detect head with PF-RPN-SIA-inspired adaptive scale selection."""

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):
        super().__init__(nc, reg_max, end2end, ch)
        self.scale_adapter = _SparseScaleAdapter(
            ch,
            hidden=int(os.getenv("YOLO_PF_SIA_HIDDEN", "64")),
            topk=int(os.getenv("YOLO_PF_SIA_TOPK", "2")),
            strength=float(os.getenv("YOLO_PF_SIA_STRENGTH", "0.5")),
        )

    def _adapted_head(self, x, box_head, cls_head):
        features, weights = self.scale_adapter(x)
        preds = super().forward_head(features, box_head, cls_head)
        preds["scale_weights"] = weights
        return preds

    def forward(self, x):
        preds = self._adapted_head(x, **self.one2many)
        if self.end2end:
            x_detach = [feature.detach() for feature in x]
            one2one = self._adapted_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)


class _CascadeFruitSelfPrompt(nn.Module):
    """Iteratively pool likely-fruit regions and feed the prompt back to features."""

    def __init__(self, channels: tuple[int, ...], hidden: int = 64, iterations: int = 2, strength: float = 0.5):
        super().__init__()
        self.iterations = max(1, int(iterations))
        self.strength = float(strength)
        self.to_prompt = nn.ModuleList(nn.Conv2d(c, hidden, 1) for c in channels)
        self.fruitness = nn.ModuleList(nn.Conv2d(c, 1, 1) for c in channels)
        self.to_gate = nn.ModuleList(nn.Conv2d(hidden, c, 1) for c in channels)
        self.prompt_seed = nn.Parameter(torch.randn(hidden) / math.sqrt(hidden))

    def forward(self, features: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]:
        enhanced = list(features)
        prompt_logits = None
        prompt = self.prompt_seed.view(1, -1)
        for _ in range(self.iterations):
            projected = [projection(feature) for projection, feature in zip(self.to_prompt, enhanced)]
            logits = [head(feature) for head, feature in zip(self.fruitness, enhanced)]
            pooled_numerator = prompt.new_zeros((features[0].shape[0], prompt.shape[-1]))
            pooled_denominator = prompt.new_zeros((features[0].shape[0], 1))
            for feature, logit in zip(projected, logits):
                mask = logit.sigmoid()
                pooled_numerator = pooled_numerator + (feature * mask).flatten(2).sum(-1)
                pooled_denominator = pooled_denominator + mask.flatten(2).sum(-1)
            visual_prompt = pooled_numerator / pooled_denominator.clamp_min(1e-6)
            prompt = F.normalize(prompt.expand_as(visual_prompt) + visual_prompt, dim=-1)
            updated = []
            for feature, projection, gate_projection in zip(enhanced, projected, self.to_gate):
                similarity = (F.normalize(projection, dim=1) * prompt[:, :, None, None]).sum(1, keepdim=True)
                prompt_feature = gate_projection(prompt[:, :, None, None])
                gate = similarity.sigmoid()
                updated.append(feature + self.strength * gate * prompt_feature)
            enhanced = updated
            prompt_logits = logits
        flattened_logits = torch.cat([logit.flatten(2) for logit in prompt_logits], dim=2)
        return enhanced, flattened_logits


class PFSelfPromptDetect(Detect):
    """FSDH head with a supervised, class-agnostic cascade self-prompt path."""

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):
        super().__init__(nc, reg_max, end2end, ch)
        self.self_prompt = _CascadeFruitSelfPrompt(
            ch,
            hidden=int(os.getenv("YOLO_PF_CSP_HIDDEN", "64")),
            iterations=int(os.getenv("YOLO_PF_CSP_ITERS", "2")),
            strength=float(os.getenv("YOLO_PF_CSP_STRENGTH", "0.5")),
        )

    def _prompted_head(self, x, box_head, cls_head):
        features, prompt_scores = self.self_prompt(x)
        preds = super().forward_head(features, box_head, cls_head)
        preds["prompt_scores"] = prompt_scores
        return preds

    def forward(self, x):
        preds = self._prompted_head(x, **self.one2many)
        if self.end2end:
            x_detach = [feature.detach() for feature in x]
            one2one = self._prompted_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def bias_init(self):
        super().bias_init()
        for head in self.self_prompt.fruitness:
            head.bias.data.fill_(-4.0)


class PFCGQSDetect(QualityDetect):
    """Maturity confidence + IoU quality + centerness joint scoring head."""

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):
        q_power = float(os.getenv("YOLO_PF_CGQS_IOU_POWER", "0.5"))
        super().__init__(nc, reg_max, end2end, ch, q_power=q_power)
        self.cls_power = float(os.getenv("YOLO_PF_CGQS_CLS_POWER", "1.0"))
        self.ctr_power = float(os.getenv("YOLO_PF_CGQS_CTR_POWER", "0.5"))
        self.cvc = nn.ModuleList()
        for channels in ch:
            hidden = max(16, channels // 2)
            self.cvc.append(nn.Sequential(Conv(channels, hidden, 3), DWConv(hidden, hidden, 3), nn.Conv2d(hidden, 1, 1)))
        if end2end:
            self.one2one_cvc = copy.deepcopy(self.cvc)

    def forward_head(self, x, box_head=None, cls_head=None, q_head=None, ctr_head=None):
        preds = super().forward_head(x, box_head, cls_head, q_head)
        if preds and ctr_head is not None:
            batch = x[0].shape[0]
            preds["centerness"] = torch.cat(
                [ctr_head[index](x[index]).view(batch, 1, -1) for index in range(self.nl)], dim=2
            )
        return preds

    def forward(self, x):
        preds = self.forward_head(x, **self.one2many, q_head=self.cvq, ctr_head=self.cvc)
        if self.end2end:
            x_detach = [feature.detach() for feature in x]
            one2one = self.forward_head(
                x_detach,
                **self.one2one,
                q_head=self.one2one_cvq,
                ctr_head=self.one2one_cvc,
            )
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x):
        boxes = self._get_decode_boxes(x)
        maturity = x["scores"].sigmoid().pow(self.cls_power)
        quality = x["quality"].sigmoid().pow(self.q_power)
        centerness = x["centerness"].sigmoid().pow(self.ctr_power)
        return torch.cat((boxes, maturity * quality * centerness), dim=1)

    def bias_init(self):
        super().bias_init()
        for branch in self.cvc:
            branch[-1].bias.data.zero_()
        if self.end2end:
            for branch in self.one2one_cvc:
                branch[-1].bias.data.zero_()


class FruitProposalMaturityDetect(Detect):
    """Class-agnostic fruit proposals followed by ROI maturity classification.

    Stage 1 predicts one fruitness score and a box at every dense location.
    Stage 2 classifies only selected proposal boxes with ROI-aligned P3 features.
    Bounding boxes are not refined in the second stage, keeping the experiment
    small enough to compare fairly with FSDH-YOLO.
    """

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):
        if end2end:
            raise ValueError("FruitProposalMaturityDetect does not support end2end/one-to-one mode")
        super().__init__(nc, reg_max, False, ch)
        # Dense maturity branches are deliberately removed: maturity is learned
        # only from ROI features after class-agnostic proposal generation.
        self.cv3 = nn.ModuleList()
        hidden = max(16, ch[0] // 2)
        self.cvobj = nn.ModuleList(
            nn.Sequential(Conv(channels, max(16, channels // 2), 3), DWConv(max(16, channels // 2), max(16, channels // 2), 3), nn.Conv2d(max(16, channels // 2), 1, 1))
            for channels in ch
        )
        self.roi_stem = nn.Sequential(Conv(ch[0], hidden, 1), Conv(hidden, hidden, 3))
        self.roi_classifier = nn.Linear(hidden, nc)
        self.roi_size = int(os.getenv("YOLO_PF_ROI_SIZE", "3"))
        self.proposal_topk = int(os.getenv("YOLO_PF_PROPOSAL_TOPK", "300"))

    @property
    def one2many(self):
        return {"box_head": self.cv2, "objectness_head": self.cvobj}

    def forward_head(self, x, box_head=None, objectness_head=None):
        if box_head is None or objectness_head is None:
            return {}
        batch = x[0].shape[0]
        boxes = torch.cat([box_head[index](x[index]).view(batch, 4 * self.reg_max, -1) for index in range(self.nl)], 2)
        fruitness = torch.cat([objectness_head[index](x[index]).view(batch, 1, -1) for index in range(self.nl)], 2)
        return {"boxes": boxes, "fruitness": fruitness, "feats": x}

    @staticmethod
    def _xywh_to_xyxy(boxes):
        center, size = boxes[..., :2], boxes[..., 2:]
        return torch.cat((center - size / 2, center + size / 2), dim=-1)

    def classify_proposals(self, features: list[torch.Tensor], boxes_by_image: list[torch.Tensor]) -> torch.Tensor:
        from torchvision.ops import roi_align

        if not boxes_by_image or sum(boxes.shape[0] for boxes in boxes_by_image) == 0:
            return features[0].new_zeros((0, self.nc))
        stride = float(self.stride[0].item())
        pooled = roi_align(
            features[0],
            boxes_by_image,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0 / stride,
            sampling_ratio=2,
            aligned=True,
        )
        pooled = self.roi_stem(pooled).mean((2, 3))
        return self.roi_classifier(pooled)

    def forward(self, x):
        preds = self.forward_head(x, **self.one2many)
        if self.training:
            return preds
        decoded = self._get_decode_boxes(preds)  # [B,4,A], xywh in image pixels
        fruitness = preds["fruitness"].sigmoid()
        k = min(self.proposal_topk, decoded.shape[-1])
        proposal_indices = fruitness.squeeze(1).topk(k, dim=1).indices
        selected_boxes = decoded.gather(2, proposal_indices[:, None, :].expand(-1, 4, -1))
        selected_fruitness = fruitness.gather(2, proposal_indices[:, None, :])
        boxes_xyxy = self._xywh_to_xyxy(selected_boxes.permute(0, 2, 1))
        image_h = x[0].shape[-2] * float(self.stride[0])
        image_w = x[0].shape[-1] * float(self.stride[0])
        boxes_by_image = []
        for boxes in boxes_xyxy:
            boxes = boxes.clone()
            boxes[:, 0::2].clamp_(0, image_w)
            boxes[:, 1::2].clamp_(0, image_h)
            boxes_by_image.append(boxes)
        roi_logits = self.classify_proposals(x, boxes_by_image)
        maturity_chunks = roi_logits.split([boxes.shape[0] for boxes in boxes_by_image], dim=0)
        maturity = torch.stack([chunk.softmax(-1).transpose(0, 1) for chunk in maturity_chunks], dim=0)
        scores = maturity * selected_fruitness
        y = torch.cat((selected_boxes, scores), dim=1)
        preds["proposal_indices"] = proposal_indices
        preds["roi_logits"] = roi_logits
        return y if self.export else (y, preds)

    def bias_init(self):
        for index, box_branch in enumerate(self.cv2):
            box_branch[-1].bias.data.fill_(2.0)
            self.cvobj[index][-1].bias.data.fill_(math.log(5 / (640 / self.stride[index]) ** 2))
        nn.init.normal_(self.roi_classifier.weight, std=0.01)
        nn.init.zeros_(self.roi_classifier.bias)
