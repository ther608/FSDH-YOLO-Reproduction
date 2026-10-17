"""Build the AP75 supplement for every quantitative configuration in paper 3.2.

The script reads the framework-neutral Common Evaluator JSON files only. It
does not run inference or training. Repeated configurations in multiple paper
tables are represented once, with all manuscript locations recorded.
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/paper_3_2_ap75_shab"


@dataclass(frozen=True)
class Spec:
    key: str
    model: str
    directory: Path
    locations: str


U = ROOT / "runs/common_eval/paper16_unified_protocol"
B = ROOT / "runs/common_eval/paper16_shab_branch_subset_and_param_match_v1"
M = ROOT / "runs/common_eval/paper16_fsdconv_shab_m_confirmation_v1"
P = ROOT / "runs/common_eval/paper16_pending_experiments"
E = ROOT / "runs/external_eval"

SPECS = [
    Spec("yolo11n", "YOLO11n", U / "yolo11n", "表4; 参数预算表; 注意力表; SHAB分支表; 消融表; 主流模型表; 近期方法表; Bootstrap表"),
    Spec("yolo11s", "YOLO11s", U / "yolo11s", "表4"),
    Spec("yolo11m", "YOLO11m", U / "yolo11m", "表4"),
    Spec("yolo11n_pm", "YOLO11n-PM", B / "YOLO11N_PM", "参数预算表"),
    Spec("attention_ca", "CA", U / "attention_ca", "注意力表"),
    Spec("attention_cbam", "CBAM", U / "attention_cbam", "注意力表"),
    Spec("attention_eca", "ECA", U / "attention_eca", "注意力表"),
    Spec("attention_ema", "EMA", U / "attention_ema", "注意力表"),
    Spec("attention_se", "SE", U / "attention_se", "注意力表"),
    Spec("attention_simam", "SimAM", U / "attention_simam", "注意力表"),
    Spec("shab", "SHAB / SHAB-LMG", U / "shab", "注意力表; SHAB分支表; 消融表"),
    Spec("shab_l", "SHAB-L", B / "SHAB_L", "SHAB分支表"),
    Spec("shab_m", "SHAB-M", B / "SHAB_M", "SHAB分支表"),
    Spec("shab_g", "SHAB-G", B / "SHAB_G", "SHAB分支表"),
    Spec("shab_lm", "SHAB-LM", B / "SHAB_LM", "SHAB分支表"),
    Spec("shab_lg", "SHAB-LG", B / "SHAB_LG", "SHAB分支表"),
    Spec("shab_mg", "SHAB-MG", B / "SHAB_MG", "SHAB分支表"),
    Spec("fsdconv_shab_m", "FSDConv+SHAB-M", M / "FSDCONV_SHAB_M", "SHAB分支表"),
    Spec("fsdconv", "+FSDConv", U / "fsdconv", "消融表"),
    Spec("fsdh", "FSDH-YOLO", U / "fsdconv_shab", "参数预算表; SHAB分支表; 消融表; 主流模型表; 近期方法表; Bootstrap表"),
    # The historical directory key is sfm_yolo_n, but the audited paper row is
    # the three-module FSDConv+SHAB+MaturityDetect negative ablation.
    Spec("fsdh_maturitydetect", "+MaturityDetect", U / "sfm_yolo_n", "消融表"),
    Spec("fsdh_pure_nwd", "+NWD Loss (pure NWD)", P / "FSD_SHAB_PURE_NWD", "消融表"),
    Spec("yolov5nu", "YOLOv5nu", U / "yolov5nu", "主流模型表"),
    Spec(
        "yolov7_tiny",
        "YOLOv7-tiny",
        E / "dsA_groupaware_clean_aug_v2__comparison__yolov7_tiny",
        "主流模型表",
    ),
    Spec("yolov8n", "YOLOv8n", U / "yolov8n", "主流模型表"),
    Spec("yolo26n", "YOLO26n", U / "yolo26n", "主流模型表"),
    Spec(
        "rtdetrv2_s",
        "RT-DETRv2-S",
        E / "dsA_groupaware_clean_aug_v2__comparison__rtdetrv2_s",
        "主流模型表",
    ),
    Spec("peanet", "PEANet-YOLOv5-D-Repro", P / "PEANet_YOLOv5_D", "近期方法表"),
    Spec("fpdw_v2", "YOLOv8-FPDW-Repro v2", P / "YOLOv8_FPDW_FIXED", "近期方法表"),
    Spec("edge_yolov11", "Edge-YOLOv11-s-Repro", P / "Edge_YOLOv11", "近期方法表"),
    Spec("gpc_yolo", "GPC-YOLOv8n-Repro", P / "GPC_YOLO", "近期方法表"),
]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    per_seed: list[dict] = []
    aggregate: list[dict] = []
    for spec in SPECS:
        values: list[float] = []
        for seed in (0, 1, 2):
            source = spec.directory / f"seed{seed}__test_metrics_common.json"
            if not source.is_file():
                raise FileNotFoundError(source)
            payload = json.loads(source.read_text(encoding="utf-8"))
            if payload.get("split") != "test":
                raise RuntimeError(f"Non-test result: {source}")
            overall = payload["overall"]
            if "ap75" not in overall:
                raise RuntimeError(f"AP75 missing: {source}")
            value = float(overall["ap75"]) * 100.0
            values.append(value)
            per_seed.append(
                {
                    "key": spec.key,
                    "model": spec.model,
                    "paper_locations": spec.locations,
                    "seed": seed,
                    "AP75_percent": value,
                    "mAP50_percent": float(overall["map50"]) * 100.0,
                    "mAP50_95_percent": float(overall["map50_95"]) * 100.0,
                    "metrics_json": str(source.resolve()),
                }
            )
        aggregate.append(
            {
                "key": spec.key,
                "model": spec.model,
                "paper_locations": spec.locations,
                "n": 3,
                "AP75_percent_mean": statistics.fmean(values),
                "AP75_percent_sample_sd": statistics.stdev(values),
                "AP75_mean_plus_minus_sd": f"{statistics.fmean(values):.3f}±{statistics.stdev(values):.3f}",
            }
        )

    by_key = {row["key"]: row for row in aggregate}
    baseline = by_key["yolo11n"]
    final = by_key["fsdh"]
    capacity = by_key["yolo11n_pm"]
    audit = {
        "paper": "论文3.2",
        "metric": "mean class AP at IoU=0.75 under the existing Common Evaluator protocol",
        "unique_configurations": len(SPECS),
        "seed_results": len(per_seed),
        "all_ap75_present": len(per_seed) == len(SPECS) * 3,
        "retraining_required": False,
        "baseline_AP75_percent_mean": baseline["AP75_percent_mean"],
        "FSDH_AP75_percent_mean": final["AP75_percent_mean"],
        "FSDH_minus_baseline_AP75_pp": final["AP75_percent_mean"] - baseline["AP75_percent_mean"],
        "FSDH_minus_YOLO11n_PM_AP75_pp": final["AP75_percent_mean"] - capacity["AP75_percent_mean"],
        "excluded": [
            "Dataset B historical/cross-dataset experiments absent from paper 3.2正文",
            "obsolete YOLOv8-FPDW result before the WIoU broadcasting fix",
            "training-time validation/test summaries outside the Common Evaluator protocol",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "AP75_逐种子.csv", per_seed)
    write_csv(OUT / "AP75_三种子汇总.csv", aggregate)
    (OUT / "AP75_审计.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
