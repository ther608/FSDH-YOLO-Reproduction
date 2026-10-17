"""Create the paper-ready Dataset B report from three-seed Common Evaluator outputs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SEEDS = (0, 1, 2)
MODELS = {
    "YOLO11n": "DatasetB4_YOLO11n",
    "FSDH-YOLO-n": "DatasetB4_FSDH_YOLO",
}
CLASSES = ("green", "half", "red", "young")
MAIN_METRICS = (
    "P",
    "R",
    "mAP50",
    "mAP50_95",
    "AP75",
    "AP_small",
    "green_AP50",
    "green_AP",
    "half_AP50",
    "half_AP",
    "red_AP50",
    "red_AP",
    "young_AP50",
    "young_AP",
)
METRIC_LABELS = {
    "AP75": "AP75",
    "green_AP": "green AP",
    "green_AP50": "green AP50",
    "half_AP": "half AP",
    "half_AP50": "half AP50",
    "red_AP": "red AP",
    "red_AP50": "red AP50",
    "young_AP": "young AP",
    "young_AP50": "young AP50",
    "AP_small": "AP-small (COCO)",
    "mAP50_95": "mAP50-95",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--latency-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-run-base", required=True)
    parser.add_argument("--fsdh-run-base", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--report-title",
        default="Dataset B v3 原生四分类跨数据集重训：Common Evaluator 结果",
    )
    parser.add_argument("--dataset-version", default="v3")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_common(payload: dict) -> dict[str, float]:
    overall = payload["overall"]
    per_class = payload["per_class"]
    quality = payload["quality"]
    row = {
        "P": overall["precision"] * 100,
        "R": overall["recall"] * 100,
        "max_F1": overall["max_mean_f1"] * 100,
        "F1_threshold": overall["confidence"],
        "mAP50": overall["map50"] * 100,
        "mAP50_95": overall["map50_95"] * 100,
        "AP75": overall["ap75"] * 100,
        "AP_small": overall["ap_small"] * 100,
        "AR300": overall["ar300"] * 100,
        "corr_score_iou_tp": quality["corr_score_iou_tp"],
        "ECE_IoU50": quality["ece_iou50"],
        "predictions": float(quality["predictions"]),
        "TP_matches": float(quality["true_positive_matches"]),
    }
    for name in CLASSES:
        row[f"{name}_AP"] = per_class[name]["ap50_95"] * 100
        row[f"{name}_AP50"] = per_class[name]["ap50"] * 100
    return row


def aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": statistics.fmean(row[key] for row in rows),
            "sample_sd": statistics.stdev(row[key] for row in rows) if len(rows) > 1 else 0.0,
        }
        for key in rows[0]
    }


def mean_sd(stats: dict[str, dict[str, float]], key: str, digits: int = 3) -> str:
    return f"{stats[key]['mean']:.{digits}f}±{stats[key]['sample_sd']:.{digits}f}"


def signed_mean_sd(stats: dict[str, dict[str, float]], key: str, digits: int = 3) -> str:
    return f"{stats[key]['mean']:+.{digits}f}±{stats[key]['sample_sd']:.{digits}f}"


def markdown_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seeds = tuple(args.seeds)
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must not contain duplicates")
    run_bases = {
        "YOLO11n": args.baseline_run_base,
        "FSDH-YOLO-n": args.fsdh_run_base,
    }
    raw: dict[str, list[dict]] = {name: [] for name in MODELS}
    rows: dict[str, list[dict[str, float]]] = {name: [] for name in MODELS}
    complexity: dict[str, list[dict[str, float | None]]] = {name: [] for name in MODELS}
    initialization: dict[str, list[dict]] = {name: [] for name in MODELS}
    protocol = None
    ground_truth = None

    for model_name, model_id in MODELS.items():
        for seed in seeds:
            common_path = args.common_root / model_id / f"seed{seed}__test_metrics_common.json"
            common = read_json(common_path)
            if common.get("seed") != seed or common.get("split") != "test":
                raise ValueError(f"Unexpected seed/split in {common_path}")
            if not Path(common["checkpoint"]).is_file():
                raise FileNotFoundError(f"Checkpoint recorded by evaluator is missing: {common['checkpoint']}")
            if protocol is None:
                protocol = common["protocol"]
                ground_truth = common.get("ground_truth")
                if protocol.get("small_area_scale") != 1.0:
                    raise ValueError("Dataset B must use standard COCO AP-small with area_scale=1")
                if protocol.get("small_metric_name") != "AP_small_COCO":
                    raise ValueError("Dataset B Common Evaluator must identify AP_small_COCO explicitly")
                if not ground_truth or ground_truth.get("images") != 212 or ground_truth.get("annotations") != 3263:
                    raise ValueError(f"Unexpected frozen Dataset B test ground truth: {ground_truth}")
            elif common["protocol"] != protocol:
                raise ValueError(f"Common Evaluator protocol mismatch: {common_path}")
            elif common.get("ground_truth") != ground_truth:
                raise ValueError(f"Common Evaluator ground-truth mismatch: {common_path}")

            test_path = args.test_root / f"{run_bases[model_name]}__seed{seed}_test" / "metrics_summary.json"
            test_payload = read_json(test_path)
            if Path(test_payload["model"]).resolve() != Path(common["checkpoint"]).resolve():
                raise ValueError(f"Test/Common checkpoint mismatch for {model_name}, seed {seed}")

            init_path = (
                args.train_root
                / f"{run_bases[model_name]}__seed{seed}"
                / "initialization_summary.json"
            )
            init_payload = read_json(init_path)
            if init_payload.get("seed_argument") != seed or not init_payload.get("deterministic"):
                raise ValueError(f"Invalid initialization manifest: {init_path}")
            if model_name == "FSDH-YOLO-n":
                if init_payload.get("weights_map") != "none":
                    raise ValueError(f"FSDH must use Dataset-A-compatible native weight transfer: {init_path}")
                if init_payload.get("mode") != "native_name_shape_transfer":
                    raise ValueError(f"FSDH native weight-transfer audit is missing: {init_path}")
            initialization[model_name].append(init_payload)

            latency = None
            if args.latency_root:
                latency_path = (
                    args.latency_root
                    / f"{run_bases[model_name]}__seed{seed}"
                    / "uniform_latency_results.json"
                )
                if latency_path.is_file():
                    latency = read_json(latency_path)["results"][0]

            row = flatten_common(common)
            rows[model_name].append(row)
            raw[model_name].append(common)
            complexity[model_name].append(
                {
                    "params_M": float(test_payload["model_complexity"]["params_m"]),
                    "gflops": float(latency["gflops"]) if latency and latency.get("gflops") else None,
                    "total_ms": float(latency["total_ms_mean"]) if latency else None,
                    "FPS": float(latency["fps_from_total_mean"]) if latency else None,
                }
            )

    stats = {name: aggregate(model_rows) for name, model_rows in rows.items()}
    paired_rows = [
        {key: rows["FSDH-YOLO-n"][i][key] - rows["YOLO11n"][i][key] for key in MAIN_METRICS}
        for i in range(len(seeds))
    ]
    paired = aggregate(paired_rows)

    complexity_stats: dict[str, dict[str, dict[str, float]]] = {}
    for model_name, model_rows in complexity.items():
        complexity_stats[model_name] = {}
        for key in ("params_M", "gflops", "total_ms", "FPS"):
            values = [float(row[key]) for row in model_rows if row[key] is not None]
            if len(values) == len(seeds):
                complexity_stats[model_name][key] = {
                    "mean": statistics.fmean(values),
                    "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                }

    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        f"# {args.report_title}",
        "",
        f"生成时间：`{generated}`  ",
        f"比较对象：YOLO11n 与 FSDH-YOLO-n（FSDConv+SHAB）；seed {'/'.join(map(str, seeds))}。  ",
        "统计口径：均值±样本标准差（ddof=1）；所有精度指标均由同一 Common Evaluator 重新计算。",
        "",
        "## 1. 统一计算协议",
        "",
        f"- split：`test`；输入尺寸：`{protocol['imgsz']}×{protocol['imgsz']}`。",
        f"- prediction confidence floor：`{protocol['prediction_confidence_floor']}`；class-aware NMS IoU：`{protocol['nms_iou']}`；max_det：`{protocol['max_det']}`。",
        f"- AP-small：`{protocol['small_metric_name']}`，采用标准 COCO small 定义（{protocol['small_area']}）；当前数据集原图均为 640×640，不使用 Dataset A 的 area×4 自定义口径。",
        f"- 冻结 test GT：{ground_truth['images']} 张、{ground_truth['annotations']} 个实例；COCO 标注 SHA-256：`{ground_truth['sha256']}`。",
        "- 四阶段按 `young → green → half → red` 解释；两个中间成熟度类别 `green` 与 `half` 的 AP 分别报告，不合并、不平均，也不使用 Dataset A 的 `semi-ripe AP` 名称。",
        "- P/R：在使四类别平均 F1 最大的置信度处计算；AP 使用完整低阈值预测集。",
        "",
        "## 2. 论文主表结果",
        "",
    ]
    lines += markdown_table(
        ("模型", "P/%", "R/%", "mAP50/%", "mAP50-95/%", "AP75/%", "AP-small COCO/%"),
        [
            (
                model_name,
                *(mean_sd(stats[model_name], key) for key in ("P", "R", "mAP50", "mAP50_95", "AP75", "AP_small")),
            )
            for model_name in MODELS
        ],
    )
    lines += ["", "## 3. 三项任务专项结果", "", "### 3.1 严格定位：AP75", ""]
    lines += markdown_table(
        ("模型", "AP75 均值±样本SD/%", "/".join(f"seed{seed}" for seed in seeds) + "/%"),
        [
            (
                model_name,
                mean_sd(stats[model_name], "AP75"),
                " / ".join(f"{row['AP75']:.3f}" for row in rows[model_name]),
            )
            for model_name in MODELS
        ],
    )
    lines += ["", "### 3.2 小目标：COCO AP-small", ""]
    lines += markdown_table(
        ("模型", "AP-small COCO 均值±样本SD/%", "/".join(f"seed{seed}" for seed in seeds) + "/%"),
        [
            (
                model_name,
                mean_sd(stats[model_name], "AP_small"),
                " / ".join(f"{row['AP_small']:.3f}" for row in rows[model_name]),
            )
            for model_name in MODELS
        ],
    )
    lines += [
        "",
        "说明：本数据集图像为 640×640，因此采用 COCO 原始 small 面积阈值 `<32²=1024 px²`；不采用 Dataset A 为密集小果实设置的 `AP_small_area4`。",
        "",
        "### 3.3 成熟度分类：各类别 AP50 与 AP50-95",
        "",
    ]
    maturity_rows = []
    for model_name in MODELS:
        for class_name in CLASSES:
            maturity_rows.append(
                (
                    model_name,
                    class_name,
                    mean_sd(stats[model_name], f"{class_name}_AP50"),
                    mean_sd(stats[model_name], f"{class_name}_AP"),
                    " / ".join(f"{row[f'{class_name}_AP50']:.3f}" for row in rows[model_name]),
                    " / ".join(f"{row[f'{class_name}_AP']:.3f}" for row in rows[model_name]),
                )
            )
    lines += markdown_table(
        ("模型", "成熟度", "AP50 均值±SD/%", "AP50-95 均值±SD/%", "逐 seed AP50/%", "逐 seed AP50-95/%"),
        maturity_rows,
    )
    lines += ["", "## 4. 逐 seed 总体结果", ""]
    per_seed_rows = []
    for model_name in MODELS:
        for seed, row in zip(seeds, rows[model_name]):
            per_seed_rows.append(
                (
                    model_name,
                    str(seed),
                    *(f"{row[key]:.3f}" for key in ("P", "R", "mAP50", "mAP50_95", "AP75", "AP_small")),
                )
            )
    lines += markdown_table(
        ("模型", "seed", "P/%", "R/%", "mAP50/%", "mAP50-95/%", "AP75/%", "AP-small COCO/%"),
        per_seed_rows,
    )
    lines += ["", "## 5. FSDH−YOLO11n 配对变化", ""]
    lines += markdown_table(
        ("指标", "配对均值±样本SD/pp", "/".join(f"seed{seed}" for seed in seeds) + "/pp"),
        [
            (
                METRIC_LABELS.get(key, key),
                signed_mean_sd(paired, key),
                " / ".join(f"{row[key]:+.3f}" for row in paired_rows),
            )
            for key in MAIN_METRICS
        ],
    )
    lines += ["", "## 6. 扩展质量指标", ""]
    extended = ("max_F1", "F1_threshold", "AR300", "corr_score_iou_tp", "ECE_IoU50", "predictions", "TP_matches")
    lines += markdown_table(
        ("模型", *extended),
        [
            (
                model_name,
                *(mean_sd(stats[model_name], key) for key in extended),
            )
            for model_name in MODELS
        ],
    )
    lines += ["", "## 7. 复杂度与统一延迟（非 Common accuracy 指标）", ""]
    complexity_rows = []
    for model_name in MODELS:
        model_stats = complexity_stats[model_name]
        complexity_rows.append(
            (
                model_name,
                mean_sd(model_stats, "params_M", 6),
                mean_sd(model_stats, "gflops") if "gflops" in model_stats else "N/A",
                mean_sd(model_stats, "total_ms") if "total_ms" in model_stats else "N/A",
                mean_sd(model_stats, "FPS") if "FPS" in model_stats else "N/A",
            )
        )
    lines += markdown_table(("模型", "Params/M", "GFLOPs", "总延迟/ms", "FPS"), complexity_rows)
    lines += [
        "",
        "## 8. 可追溯性",
        "",
        f"- Common Evaluator 根目录：`{args.common_root.resolve()}`",
        f"- 训练与初始化清单根目录：`{args.train_root.resolve()}`",
        f"- 独立 test 根目录：`{args.test_root.resolve()}`",
        f"- 统一 latency 根目录：`{args.latency_root.resolve() if args.latency_root else '未提供'}`",
        "- 本次 Dataset B 双模型对比中，FSDH 仅使用 Ultralytics 原生同名/同形状权重迁移，不做语义层索引重映射。",
        "- 每个 seed 的建模前种子、原生权重迁移覆盖率记录在 `initialization_summary.json`。",
        "- 每个指标均可由对应 `seed{k}__test_metrics_common.json` 与 `seed{k}__test_predictions_coco.json` 回溯。",
        "- 本文件只报告数值与配对方向；机制解释和统计显著性结论应在人工复核后写入论文。",
        "",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    task_ap75_csv = args.output.with_name(f"{args.output.stem}_task_ap75.csv")
    task_apsmall_csv = args.output.with_name(f"{args.output.stem}_task_apsmall_coco.csv")
    task_maturity_csv = args.output.with_name(f"{args.output.stem}_task_maturity_ap.csv")
    write_csv(
        task_ap75_csv,
        ("model", "seed", "AP75_percent"),
        [
            {"model": model_name, "seed": seed, "AP75_percent": rows[model_name][index]["AP75"]}
            for model_name in MODELS for index, seed in enumerate(seeds)
        ],
    )
    write_csv(
        task_apsmall_csv,
        ("model", "seed", "metric", "area_definition", "AP_small_percent"),
        [
            {
                "model": model_name,
                "seed": seed,
                "metric": protocol["small_metric_name"],
                "area_definition": protocol["small_area"],
                "AP_small_percent": rows[model_name][index]["AP_small"],
            }
            for model_name in MODELS for index, seed in enumerate(seeds)
        ],
    )
    write_csv(
        task_maturity_csv,
        ("model", "seed", "class", "AP50_percent", "AP50_95_percent"),
        [
            {
                "model": model_name,
                "seed": seed,
                "class": class_name,
                "AP50_percent": rows[model_name][index][f"{class_name}_AP50"],
                "AP50_95_percent": rows[model_name][index][f"{class_name}_AP"],
            }
            for model_name in MODELS for index, seed in enumerate(seeds) for class_name in CLASSES
        ],
    )
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "schema": f"dataset_b_{args.dataset_version}_comparison_report_v1",
                "generated_at": generated,
                "protocol": protocol,
                "ground_truth": ground_truth,
                "models": {
                    name: {
                        "rows": rows[name],
                        "aggregate": stats[name],
                        "initialization": initialization[name],
                        "complexity": complexity[name],
                        "complexity_aggregate": complexity_stats[name],
                    }
                    for name in MODELS
                },
                "paired_fsdh_minus_yolo11n": {"rows": paired_rows, "aggregate": paired},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {sidecar}")
    print(f"Wrote {task_ap75_csv}")
    print(f"Wrote {task_apsmall_csv}")
    print(f"Wrote {task_maturity_csv}")


if __name__ == "__main__":
    main()
