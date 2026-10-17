"""Audit and summarize paper16 metrics under one common COCO evaluator."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMON_ROOT = ROOT / "runs/common_eval/paper16_unified_protocol"
TRAIN_ROOT = ROOT / "runs/detect/paper_groupaware_clean_v2_reruns"
QUALITY_ROOT = ROOT / "runs/quality/paper_groupaware_clean_v2_reruns"
OUT = ROOT / "analysis/paper16_unified_common_protocol"

SPECS = [
    ("yolo11n", "YOLO11n", "dsA_groupaware_clean_aug_v2__scale__yolo11n"),
    ("yolo11s", "YOLO11s", "dsA_groupaware_clean_aug_v2__scale__yolo11s"),
    ("yolo11m", "YOLO11m", "dsA_groupaware_clean_aug_v2__scale__yolo11m"),
    ("attention_ca", "CA", "dsA_groupaware_clean_aug_v2__attention__ca"),
    ("attention_cbam", "CBAM", "dsA_groupaware_clean_aug_v2__attention__cbam"),
    ("attention_eca", "ECA", "dsA_groupaware_clean_aug_v2__attention__eca"),
    ("attention_ema", "EMA", "dsA_groupaware_clean_aug_v2__attention__ema"),
    ("attention_se", "SE", "dsA_groupaware_clean_aug_v2__attention__se"),
    ("attention_simam", "SimAM", "dsA_groupaware_clean_aug_v2__attention__simam"),
    ("shab", "SHAB", "dsA_groupaware_clean_aug_v2__ablation__shab"),
    ("fsdconv", "FSDConv", "dsA_groupaware_clean_aug_v2__fsd_completion__fsd_only"),
    ("maturitydetect", "MaturityDetect", "dsA_groupaware_clean_aug_v2__ablation__maturitydetect"),
    ("shab_maturitydetect", "SHAB+MaturityDetect", "dsA_groupaware_clean_aug_v2__ablation__maturitydetect_shab"),
    ("fsdconv_shab", "FSDConv+SHAB", "dsA_groupaware_clean_aug_v2__fsd_completion__fsd_shab"),
    ("fsdconv_maturitydetect", "FSDConv+MaturityDetect", "dsA_groupaware_clean_aug_v2__fsd_completion__fsd_maturitydetect"),
    ("sfm_yolo_n", "SFM-YOLO-n", "dsA_groupaware_clean_aug_v2__spd_position__spo_stem"),
    ("candidate_nwd025_l005", "NWD-0.25 lambda_m=0.05", "dsA_groupaware_clean_aug_v2__careplacement__nwd"),
    ("candidate_dysample", "DySample", "dsA_groupaware_clean_aug_v2__careplacement__dysample"),
    ("candidate_pure_nwd", "pure NWD lambda_m=0.10", "dsA_groupaware_clean_aug_v2__finalloss__md_shab_pure_nwd"),
    ("candidate_nwd025_l010", "NWD-0.25 lambda_m=0.10", "dsA_groupaware_clean_aug_v2__finalloss__md_shab_ciou075_nwd025"),
    ("candidate_lite_bifpn", "LiteBiFPN", "dsA_groupaware_clean_aug_v2__thirdmodule__md_shab_lite_bifpn"),
    ("candidate_p2", "P2", "dsA_groupaware_clean_aug_v2__thirdmodule__md_shab_p2"),
    ("candidate_gcdloss", "GCD loss", "dsA_groupaware_clean_aug_v2__thirdmodule__md_shab_gcdloss"),
    ("control_no_ca_l005", "MD+SHAB CIoU lambda_m=0.05 control", "dsA_groupaware_clean_aug_v2__careplacement__no_ca"),
    ("control_ciou_l010", "MD+SHAB CIoU lambda_m=0.10 control", "dsA_groupaware_clean_aug_v2__thirdmodule_confirmation__md_shab_ciou_l010"),
    ("yolov5nu", "YOLOv5nu", "dsA_groupaware_clean_aug_v2__comparison__yolov5nu"),
    ("yolov8n", "YOLOv8n", "dsA_groupaware_clean_aug_v2__comparison__yolov8n"),
    ("yolo26n", "YOLO26n", "dsA_groupaware_clean_aug_v2__comparison__yolo26n"),
]

EXTERNAL = [
    ("yolov7_tiny", "YOLOv7-tiny", ROOT / "runs/external_eval/dsA_groupaware_clean_aug_v2__comparison__yolov7_tiny"),
    ("rtdetrv2_s", "RT-DETRv2-S", ROOT / "runs/external_eval/dsA_groupaware_clean_aug_v2__comparison__rtdetrv2_s"),
]


def flatten(key: str, label: str, seed: int, payload: dict, source: Path, run_base: str = "") -> dict:
    protocol = payload["protocol"]
    if payload.get("split") != "test":
        raise RuntimeError(f"Non-test result: {source}")
    expected = {"imgsz": 640, "prediction_confidence_floor": 0.001, "max_det": 300}
    for name, value in expected.items():
        if protocol.get(name) != value:
            raise RuntimeError(f"Protocol mismatch {name}: {source}")
    if "nms_iou" in protocol and protocol["nms_iou"] != 0.7:
        raise RuntimeError(f"Protocol mismatch nms_iou: {source}")
    overall = payload["overall"]
    return {
        "key": key,
        "model": label,
        "run_base": run_base,
        "seed": seed,
        "P_percent": float(overall["precision"]) * 100,
        "R_percent": float(overall["recall"]) * 100,
        "mAP50_percent": float(overall["map50"]) * 100,
        "AP75_percent": float(overall["ap75"]) * 100,
        "mAP50_95_percent": float(overall["map50_95"]) * 100,
        "semi_ripe_AP50_95_percent": float(payload["per_class"]["semi-ripe"]["ap50_95"]) * 100,
        "AP_small_area4_percent": float(overall["ap_small_area4"]) * 100,
        "metrics_json": str(source.resolve()),
        "checkpoint": str(payload.get("checkpoint", "")),
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values)


def write_rows(name: str, rows: list[dict]) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows: list[dict] = []
    ap_small_diffs: list[float] = []
    for key, label, run_base in SPECS:
        for seed in (0, 1, 2):
            source = COMMON_ROOT / key / f"seed{seed}__test_metrics_common.json"
            if not source.is_file():
                raise FileNotFoundError(f"Missing common result: {source}")
            payload = json.loads(source.read_text(encoding="utf-8"))
            expected_checkpoint = (TRAIN_ROOT / f"{run_base}__seed{seed}/weights/best.pt").resolve()
            if Path(payload.get("checkpoint", "")).resolve() != expected_checkpoint:
                raise RuntimeError(f"Checkpoint mismatch: {source}")
            row = flatten(key, label, seed, payload, source, run_base)
            quality_path = QUALITY_ROOT / f"{run_base}__seed{seed}_test_quality.json"
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            if quality.get("size_ap_protocol", {}).get("name") != "COCOeval area-range ignore semantics":
                raise RuntimeError(f"Stale AP-small protocol: {quality_path}")
            diff = abs(row["AP_small_area4_percent"] - float(quality["ap_small_50_95"]) * 100)
            ap_small_diffs.append(diff)
            row["quality_json"] = str(quality_path.resolve())
            row["ap_small_common_vs_quality_abs_diff_pp"] = diff
            rows.append(row)

    for key, label, directory in EXTERNAL:
        for seed in (0, 1, 2):
            source = directory / f"seed{seed}__test_metrics_common.json"
            rows.append(flatten(key, label, seed, json.loads(source.read_text(encoding="utf-8")), source))

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["key"], []).append(row)
    aggregate: list[dict] = []
    metrics = [
        "P_percent",
        "R_percent",
        "mAP50_percent",
        "AP75_percent",
        "mAP50_95_percent",
        "semi_ripe_AP50_95_percent",
        "AP_small_area4_percent",
    ]
    for key, group in grouped.items():
        group.sort(key=lambda item: item["seed"])
        if [item["seed"] for item in group] != [0, 1, 2]:
            raise RuntimeError(f"Incomplete seeds: {key}")
        item = {"key": key, "model": group[0]["model"], "run_base": group[0]["run_base"], "n": 3}
        for metric in metrics:
            mean, sd = mean_sd([float(row[metric]) for row in group])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_sample_sd"] = sd
        item["metrics_paths"] = " | ".join(row["metrics_json"] for row in group)
        aggregate.append(item)

    OUT.mkdir(parents=True, exist_ok=True)
    write_rows("per_seed", rows)
    write_rows("three_seed_mean_sd", aggregate)
    audit = {
        "protocol": {
            "dataset": "Dataset A current group-aware cleaned v2 test split",
            "evaluator": "single framework-neutral pycocotools COCO evaluator",
            "imgsz": 640,
            "rect": False,
            "prediction_confidence_floor": 0.001,
            "nms_iou": 0.7,
            "max_det": 300,
            "coco_max_dets": [1, 10, 300],
            "semi_ripe_metric": "class-wise COCO AP at IoU=0.50:0.95",
            "AP75": "mean class AP at IoU=0.75",
            "AP_small_area4": "COCO area-range ignore, original-image area < 4096 px^2",
        },
        "internal_model_seed_results": len(SPECS) * 3,
        "external_model_seed_results": len(EXTERNAL) * 3,
        "three_seed_model_groups": len(aggregate),
        "max_internal_ap_small_common_vs_quality_abs_diff_pp": max(ap_small_diffs),
        # The two adapters may differ below 0.001 percentage points because one
        # predicts in batches and the common adapter streams images. This is far
        # below the paper's three-decimal reporting precision and is not a
        # protocol difference.
        "ap_small_consistency_tolerance_pp": 0.001,
        "ap_small_consistent": max(ap_small_diffs) <= 0.001,
    }
    (OUT / "protocol_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
