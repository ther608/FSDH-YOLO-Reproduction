import argparse
import csv
import json
import os
import sys
from multiprocessing import freeze_support
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import ultralytics
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument("--name", required=True, help="Experiment name for output folder")
    parser.add_argument(
        "--project",
        default=str(ROOT / "runs" / "test" / "lychee"),
        help="Project directory for val results",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--soft-nms", action="store_true", default=False)
    parser.add_argument("--soft-nms-method", default="linear", choices=["linear", "gaussian"])
    parser.add_argument("--soft-nms-sigma", type=float, default=0.5)
    parser.add_argument(
        "--save-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save JSON/CSV metric summaries",
    )
    return parser.parse_args()


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _class_metric(values, index):
    try:
        return float(values[index])
    except (IndexError, TypeError):
        return None


def _parameter_count(model):
    """Return the exact number of parameters in a loaded YOLO model."""
    nn_model = getattr(model, "model", model)
    return int(sum(parameter.numel() for parameter in nn_model.parameters()))


def save_metric_summary(metrics, model, model_path, args):
    save_dir = Path(getattr(metrics, "save_dir", Path(args.project) / args.name))
    save_dir.mkdir(parents=True, exist_ok=True)

    box = metrics.box
    speed = getattr(metrics, "speed", {}) or {}
    inference_ms = _to_float(speed.get("inference"))
    end_to_end_ms = sum(_to_float(speed.get(k)) or 0.0 for k in ("preprocess", "inference", "postprocess"))
    parameter_count = _parameter_count(model)

    names = getattr(metrics, "names", {}) or {}
    class_ids = [int(c) for c in getattr(box, "ap_class_index", [])]
    per_class = []
    for i, class_id in enumerate(class_ids):
        per_class.append(
            {
                "class_id": class_id,
                "name": names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id),
                "images": int(metrics.nt_per_image[class_id]) if hasattr(metrics, "nt_per_image") else None,
                "instances": int(metrics.nt_per_class[class_id]) if hasattr(metrics, "nt_per_class") else None,
                "precision": _class_metric(box.p, i),
                "recall": _class_metric(box.r, i),
                "map50": _class_metric(box.ap50, i),
                "map50_95": _class_metric(box.ap, i),
            }
        )

    summary = {
        "model": str(model_path),
        "data": args.data,
        "split": "test",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "soft_nms": args.soft_nms,
        "soft_nms_method": args.soft_nms_method,
        "soft_nms_sigma": args.soft_nms_sigma,
        "model_complexity": {
            "parameters": parameter_count,
            "params_m": parameter_count / 1_000_000.0,
        },
        "overall": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "speed_ms_per_image": {
            "preprocess": _to_float(speed.get("preprocess")),
            "inference": inference_ms,
            "loss": _to_float(speed.get("loss")),
            "postprocess": _to_float(speed.get("postprocess")),
            "end_to_end_no_loss": end_to_end_ms,
        },
        "fps": {
            "inference_only": (1000.0 / inference_ms) if inference_ms and inference_ms > 0 else None,
            "preprocess_inference_postprocess": (1000.0 / end_to_end_ms) if end_to_end_ms > 0 else None,
        },
        "per_class": per_class,
        "results_dict": {k: _to_float(v) for k, v in getattr(metrics, "results_dict", {}).items()},
    }

    json_path = save_dir / "metrics_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = save_dir / "per_class_metrics.csv"
    # utf-8-sig keeps the CSV readable when opened directly by Microsoft Excel.
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["class_id", "name", "images", "instances", "precision", "recall", "map50", "map50_95"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_class)

    # Save a concise, progress-free text report. This replaces captured console
    # logs, which turn carriage-return progress updates into hundreds of lines.
    text_path = save_dir / "test_results.txt"
    text_lines = [
        f"model={model_path}",
        f"data={args.data}",
        "split=test",
        f"P={summary['overall']['precision']}",
        f"R={summary['overall']['recall']}",
        f"mAP50={summary['overall']['map50']}",
        f"mAP50-95={summary['overall']['map50_95']}",
        f"parameters={parameter_count}",
        f"Params/M={parameter_count / 1_000_000.0}",
        f"FPS_inference_only={summary['fps']['inference_only']}",
        f"FPS_end_to_end={summary['fps']['preprocess_inference_postprocess']}",
    ]
    for row in per_class:
        text_lines.append(
            "class={name} P={precision} R={recall} mAP50={map50} mAP50-95={map50_95}".format(**row)
        )
    text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8-sig")

    print(f"Metric summary saved to {json_path}")
    print(f"Per-class metrics saved to {csv_path}")
    print(f"Concise test results saved to {text_path}")


def main():
    args = parse_args()
    # Soft-NMS env vars (read by non_max_suppression)
    os.environ["YOLO_SOFT_NMS"] = "1" if args.soft_nms else "0"
    os.environ["YOLO_SOFT_NMS_METHOD"] = args.soft_nms_method
    os.environ["YOLO_SOFT_NMS_SIGMA"] = str(args.soft_nms_sigma)

    print(f"ultralytics file = {ultralytics.__file__}")
    if args.soft_nms:
        print(f"Soft-NMS: method={args.soft_nms_method}, sigma={args.soft_nms_sigma}")

    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        split="test",
        project=args.project,
        name=args.name,
        workers=0,
        exist_ok=False,
    )

    print("Validation complete")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"Precision: {metrics.box.mp}")
    print(f"Recall: {metrics.box.mr}")
    if args.save_summary:
        save_metric_summary(metrics, model, args.model, args)


if __name__ == "__main__":
    freeze_support()
    main()
