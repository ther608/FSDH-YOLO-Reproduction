"""Evaluate framework-neutral COCO predictions with one paper protocol.

The same script is used across internal and external detectors. It reports COCO
AP, macro P/R at the confidence point maximizing mean class F1, configurable
original-image AP-small, per-class AP, score-IoU correlation, and detection ECE.
Predictions must use COCO xywh boxes and confidence scores.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def iou_xywh(left: list[float], right: list[float]) -> float:
    lx1, ly1, lw, lh = left
    rx1, ry1, rw, rh = right
    lx2, ly2, rx2, ry2 = lx1 + lw, ly1 + lh, rx1 + rw, ry1 + rh
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(0.0, min(ly2, ry2) - max(ly1, ry1))
    union = max(lw * lh + rw * rh - intersection, 1e-12)
    return intersection / union


def normalize_predictions(gt: dict, raw: list[dict]) -> list[dict]:
    stem_to_id = {Path(item["file_name"]).stem: int(item["id"]) for item in gt["images"]}
    filename_to_id = {item["file_name"]: int(item["id"]) for item in gt["images"]}
    valid_ids = {int(item["id"]) for item in gt["images"]}
    valid_categories = {int(item["id"]) for item in gt["categories"]}
    output = []
    for item in raw:
        image_id = item["image_id"]
        if isinstance(image_id, str):
            image_id = filename_to_id.get(image_id, stem_to_id.get(image_id, stem_to_id.get(Path(image_id).stem)))
        image_id = int(image_id)
        if image_id not in valid_ids:
            raise ValueError(f"Prediction references unknown image_id={item['image_id']!r}")
        category_id = int(item["category_id"])
        if category_id not in valid_categories:
            raise ValueError(f"Prediction references unknown category_id={category_id}")
        output.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(value) for value in item["bbox"]],
                "score": float(item["score"]),
            }
        )
    return output


def coco_metrics(annotation_path: Path, predictions: list[dict], area_scale: float = 4.0) -> tuple[dict, dict]:
    coco_gt = COCO(str(annotation_path))
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.maxDets = [1, 10, 300]
    # Dataset A uses area_scale=4; native Dataset B uses standard COCO scale=1.
    small_limit = 32**2 * area_scale
    evaluator.params.areaRng = [[0, 1e10], [0, small_limit], [small_limit, 96**2], [96**2, 1e10]]
    # Keep COCO's required labels; only the numerical "small" range is changed.
    evaluator.params.areaRngLbl = ["all", "small", "medium", "large"]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    precision = evaluator.eval["precision"]
    stats = evaluator.stats
    category_ids = evaluator.params.catIds
    names = {item["id"]: item["name"] for item in coco_gt.loadCats(category_ids)}
    per_class = {}
    for index, category_id in enumerate(category_ids):
        ap = precision[:, :, index, 0, -1]
        ap50 = precision[0, :, index, 0, -1]
        per_class[names[category_id]] = {
            "ap50_95": float(ap[ap > -1].mean()),
            "ap50": float(ap50[ap50 > -1].mean()),
        }
    # COCOeval._summarizeDets() hard-codes maxDets=100 for stats[0]. Our paper
    # protocol intentionally uses maxDets=300 because images contain dense fruit,
    # so stats[0] becomes -1 even though the accumulated precision tensor is
    # valid. Compute the all-class AP directly at the configured final maxDet.
    ap_all = precision[:, :, :, 0, -1]
    small_ap = float(stats[3])
    overall = {
        "map50_95": float(ap_all[ap_all > -1].mean()),
        "map50": float(stats[1]),
        "ap75": float(stats[2]),
        "ap_small": small_ap,
        "ar300": float(stats[8]),
    }
    if math.isclose(area_scale, 4.0):
        overall["ap_small_area4"] = small_ap  # compatibility key used by existing Dataset-A summaries
    return overall, per_class


def match_iou50(gt: dict, predictions: list[dict]) -> list[dict]:
    ground_truth: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for item in gt["annotations"]:
        ground_truth[(int(item["image_id"]), int(item["category_id"]))].append(item)
    used: dict[tuple[int, int], set[int]] = defaultdict(set)
    rows = []
    for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
        key = (prediction["image_id"], prediction["category_id"])
        candidates = ground_truth.get(key, [])
        best_index, best_iou = -1, 0.0
        for index, target in enumerate(candidates):
            if index in used[key]:
                continue
            value = iou_xywh(prediction["bbox"], target["bbox"])
            if value > best_iou:
                best_index, best_iou = index, value
        correct = best_index >= 0 and best_iou >= 0.5
        if correct:
            used[key].add(best_index)
        rows.append({**prediction, "correct_iou50": correct, "matched_iou": best_iou if correct else 0.0})
    return rows


def operating_point(gt: dict, matches: list[dict]) -> dict:
    category_ids = sorted(int(item["id"]) for item in gt["categories"])
    gt_counts = {category_id: 0 for category_id in category_ids}
    for item in gt["annotations"]:
        gt_counts[int(item["category_id"])] += 1
    thresholds = np.linspace(0.0, 1.0, 1001)
    class_precision, class_recall = [], []
    for category_id in category_ids:
        rows = [item for item in matches if item["category_id"] == category_id]
        scores = np.asarray([item["score"] for item in rows], dtype=np.float64)
        correct = np.asarray([item["correct_iou50"] for item in rows], dtype=bool)
        precision, recall = [], []
        for threshold in thresholds:
            selected = scores >= threshold
            tp = int(correct[selected].sum())
            fp = int(selected.sum()) - tp
            precision.append(tp / max(tp + fp, 1))
            recall.append(tp / max(gt_counts[category_id], 1))
        class_precision.append(precision)
        class_recall.append(recall)
    p = np.asarray(class_precision)
    r = np.asarray(class_recall)
    mean_p, mean_r = p.mean(axis=0), r.mean(axis=0)
    f1 = 2 * mean_p * mean_r / np.maximum(mean_p + mean_r, 1e-12)
    index = int(np.nanargmax(f1))
    return {
        "precision": float(mean_p[index]),
        "recall": float(mean_r[index]),
        "max_mean_f1": float(f1[index]),
        "confidence": float(thresholds[index]),
        "definition": "macro class P/R at confidence maximizing mean class F1, class-aware IoU>=0.5",
    }


def diagnostics(matches: list[dict], bins: int) -> dict:
    scores = np.asarray([item["score"] for item in matches], dtype=np.float64)
    correctness = np.asarray([item["correct_iou50"] for item in matches], dtype=np.float64)
    matched = [item for item in matches if item["correct_iou50"]]
    if len(matched) >= 2:
        matched_scores = np.asarray([item["score"] for item in matched], dtype=np.float64)
        matched_ious = np.asarray([item["matched_iou"] for item in matched], dtype=np.float64)
        corr = float(np.corrcoef(matched_scores, matched_ious)[0, 1])
    else:
        corr = math.nan
    ece = 0.0
    for left, right in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (scores >= left) & (scores < right if right < 1 else scores <= right)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correctness[mask].mean()) - float(scores[mask].mean()))
    return {
        "corr_score_iou_tp": corr,
        "ece_iou50": ece,
        "ece_bins": bins,
        "predictions": len(matches),
        "true_positive_matches": len(matched),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--area-scale", type=float, default=4.0)
    args = parser.parse_args()
    if args.area_scale <= 0:
        raise ValueError("--area-scale must be positive")
    gt = json.loads(args.annotations.read_text(encoding="utf-8"))
    raw = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions = normalize_predictions(gt, raw)
    overall, per_class = coco_metrics(args.annotations, predictions, args.area_scale)
    matches = match_iou50(gt, predictions)
    summary = {
        "schema": "external_detection_metrics_v2",
        "model": args.model,
        "seed": args.seed,
        "annotations": str(args.annotations.resolve()),
        "predictions": str(args.predictions.resolve()),
        "split": "test",
        "protocol": {
            "imgsz": 640,
            "prediction_confidence_floor": 0.001,
            "max_det": 300,
            "coco_max_dets": [1, 10, 300],
            "small_area_scale": args.area_scale,
            "small_area": f"original-image area < {32**2 * args.area_scale:g} pixels",
        },
        "overall": {**operating_point(gt, matches), **overall},
        "per_class": per_class,
        "quality": diagnostics(matches, args.bins),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
