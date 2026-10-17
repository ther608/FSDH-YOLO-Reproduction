"""Evaluate an Ultralytics checkpoint with the paper's framework-neutral common evaluator.

This adapter exports predictions in COCO xywh format, then delegates every metric
calculation (including the explicitly named AP-small area protocol) to
evaluate_coco_predictions_full_metrics.py.
It exists so internal Ultralytics models and external YOLOv7/RT-DETR models share
the same ground truth, area ranges, max_det=300, and metric implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from evaluate_coco_predictions_full_metrics import (
    coco_metrics,
    diagnostics,
    match_iou50,
    normalize_predictions,
    operating_point,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--area-scale", type=float, default=4.0)
    return parser.parse_args()


def collect_predictions(model: YOLO, args: argparse.Namespace, ground_truth: dict) -> list[dict]:
    sources = [str(args.images / item["file_name"]) for item in ground_truth["images"]]
    missing = [source for source in sources if not Path(source).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} test images; first missing file: {missing[0]}")

    predictions: list[dict] = []
    results = model.predict(
        source=sources,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        rect=False,
        stream=True,
        verbose=False,
    )
    for image_record, result in zip(ground_truth["images"], results):
        if result.boxes is None:
            continue
        image_id = int(image_record["id"])
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, class_id in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(float, box)
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(class_id),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    return predictions


def main() -> None:
    args = parse_args()
    if args.area_scale <= 0:
        raise ValueError("--area-scale must be positive")
    if not args.model.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.model}")
    if not args.annotations.is_file():
        raise FileNotFoundError(f"Missing annotations: {args.annotations}")

    ground_truth = json.loads(args.annotations.read_text(encoding="utf-8"))
    if ground_truth.get("info", {}).get("split") != "test":
        raise ValueError(f"Common Evaluator requires a test annotation file: {args.annotations}")
    image_names = [item["file_name"] for item in ground_truth["images"]]
    if len(image_names) != len(set(image_names)):
        raise ValueError("Duplicate image filenames found in Common Evaluator ground truth")
    raw_predictions = collect_predictions(YOLO(str(args.model)), args, ground_truth)
    predictions = normalize_predictions(ground_truth, raw_predictions)

    predictions_out = args.predictions_out or args.out.with_name(
        args.out.name.replace("__test_metrics_common.json", "__test_predictions_coco.json")
    )
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    predictions_out.write_text(
        json.dumps(predictions, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    overall, per_class = coco_metrics(args.annotations, predictions, args.area_scale)
    matches = match_iou50(ground_truth, predictions)
    summary = {
        "schema": "external_detection_metrics_v2",
        "model": args.model_name,
        "seed": args.seed,
        "checkpoint": str(args.model.resolve()),
        "annotations": str(args.annotations.resolve()),
        "predictions": str(predictions_out.resolve()),
        "split": "test",
        "ground_truth": {
            "sha256": sha256(args.annotations),
            "images": len(ground_truth["images"]),
            "annotations": len(ground_truth["annotations"]),
            "categories": [item["name"] for item in ground_truth["categories"]],
        },
        "protocol": {
            "imgsz": args.imgsz,
            "prediction_confidence_floor": args.conf,
            "nms_iou": args.iou,
            "max_det": args.max_det,
            "coco_max_dets": [1, 10, 300],
            "small_area_scale": args.area_scale,
            "small_area": f"original-image area < {32**2 * args.area_scale:g} pixels",
            "small_metric_name": "AP_small_COCO" if args.area_scale == 1.0 else f"AP_small_area{args.area_scale:g}",
        },
        "overall": {**operating_point(ground_truth, matches), **overall},
        "per_class": per_class,
        "quality": diagnostics(matches, args.bins),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
