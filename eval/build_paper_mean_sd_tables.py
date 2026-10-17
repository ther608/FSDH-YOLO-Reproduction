"""Create manuscript-ready mean ± sample-SD tables from audited metrics."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


SUITE_TITLES = {
    "scale_full": "Dataset A 尺度选择",
    "attention": "Dataset A 注意力机制比较",
    "comparison": "Dataset A 原生框架模型比较",
    "ablation": "Dataset A 消融实验",
    "cross_dataset": "Dataset B 独立数据集重复验证",
}

METRICS = [
    ("P/%", 3),
    ("R/%", 3),
    ("mAP@0.5/%", 3),
    ("mAP@0.5-0.95/%", 3),
    ("AP_small_area4", 4),
    ("AP_small", 4),
    ("semi-ripe AP50-95", 4),
    ("Corr", 4),
    ("ECE", 4),
]


def format_pair(mean: str, sd: str, digits: int) -> str:
    if mean in {"", None}:
        return "—"
    return f"{float(mean):.{digits}f} ± {float(sd):.{digits}f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def external_framework_rows(quality_root: Path) -> list[dict]:
    rows = []
    rtdetr_path = quality_root / "rtdetrv2_s_three_seed_test_metrics.csv"
    with rtdetr_path.open(newline="", encoding="utf-8-sig") as handle:
        data = list(csv.DictReader(handle))
    mappings = {
        "mAP@0.5/%": ("AP50", 100.0),
        "mAP@0.5-0.95/%": ("AP50_95", 100.0),
        "AP_small": ("AP_small", 1.0),
        "semi-ripe AP50-95": ("semi-ripe_AP50_95", 1.0),
    }
    row = {"Model": "RT-DETRv2-S", "n": len(data), "P/%": "—", "R/%": "—", "AP_small_area4": "—", "Corr": "—", "ECE": "—"}
    for output, (source, scale) in mappings.items():
        values = [float(item[source]) * scale for item in data]
        digits = 3 if "/%" in output else 4
        row[output] = f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"
    rows.append(row)

    yolov7_path = quality_root / "yolov7_tiny_three_seed_test_metrics.csv"
    with yolov7_path.open(newline="", encoding="utf-8-sig") as handle:
        data = list(csv.DictReader(handle))
    mappings = {
        "P/%": ("P", 1.0),
        "R/%": ("R", 1.0),
        "mAP@0.5/%": ("mAP50", 1.0),
        "mAP@0.5-0.95/%": ("mAP50_95", 1.0),
        "AP_small_area4": ("AP_small_area4", 1.0),
        "semi-ripe AP50-95": ("semi-ripe AP50-95", 1.0),
        "Corr": ("Corr", 1.0),
        "ECE": ("ECE", 1.0),
    }
    row = {"Model": "YOLOv7-tiny", "n": len(data), "AP_small": "—"}
    for output, (source, scale) in mappings.items():
        values = [float(item[source]) * scale for item in data]
        digits = 3 if "/%" in output else 4
        row[output] = f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"
    rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with args.statistics.open(newline="", encoding="utf-8-sig") as handle:
        source = list(csv.DictReader(handle))

    markdown = [
        "# 论文实验均值 ± 样本标准差表",
        "",
        "所有随机性指标均由 seed=0、1、2 三次独立训练计算，标准差为样本标准差（分母 n-1）。",
        "参数量和 FLOPs 是模型常数，不附标准差。旧 FPS 不作为最终统一时延结论。",
        "",
    ]
    fields = ["Model", "n"] + [name for name, _ in METRICS]
    for suite, title in SUITE_TITLES.items():
        rows = []
        for item in source:
            if item["Suite"] != suite:
                continue
            row = {"Model": item["Model"], "n": item["n"]}
            for metric, digits in METRICS:
                row[metric] = format_pair(item.get(f"{metric} mean", ""), item.get(f"{metric} sd", ""), digits)
            rows.append(row)
        if not rows:
            continue
        write_csv(args.out / f"{suite}_mean_sd.csv", rows)
        markdown.extend([f"## {title}", "", "| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"])
        for row in rows:
            markdown.append("| " + " | ".join(str(row.get(field, "—")) for field in fields) + " |")
        markdown.append("")

    external = external_framework_rows(args.quality_root)
    write_csv(args.out / "external_framework_mean_sd.csv", external)
    markdown.extend(
        [
            "## Dataset A 外部框架参考",
            "",
            "RT-DETRv2-S 的 P/R、Corr、ECE 与 Ultralytics 口径未统一，保留为空；其标准 COCO AP_small "
            "不能与 Dataset A 的 AP_small_area4 直接比较。",
            "",
            "| " + " | ".join(fields) + " |",
            "|" + "|".join(["---"] * len(fields)) + "|",
        ]
    )
    for row in external:
        markdown.append("| " + " | ".join(str(row.get(field, "—")) for field in fields) + " |")
    markdown.append("")

    combined_models = {"YOLOv5nu", "YOLOv8n", "YOLO26n", "YOLO11n", "MSCA-YOLO-n"}
    combined = []
    for item in source:
        if item["Model"] not in combined_models:
            continue
        if item["Model"] in {"YOLO11n", "MSCA-YOLO-n"} and item["Suite"] != "scale_full":
            continue
        if item["Model"] not in {"YOLO11n", "MSCA-YOLO-n"} and item["Suite"] != "comparison":
            continue
        row = {"Model": item["Model"], "n": item["n"]}
        for metric, digits in METRICS:
            row[metric] = format_pair(item.get(f"{metric} mean", ""), item.get(f"{metric} sd", ""), digits)
        combined.append(row)
    combined.extend(external)
    order = {
        name: index
        for index, name in enumerate(
            ["YOLOv5nu", "YOLOv7-tiny", "YOLOv8n", "YOLO11n", "YOLO26n", "RT-DETRv2-S", "MSCA-YOLO-n"]
        )
    }
    combined.sort(key=lambda row: order[row["Model"]])
    write_csv(args.out / "model_comparison_combined_mean_sd.csv", combined)
    markdown.extend(
        [
            "## Dataset A 模型比较合并表",
            "",
            "| " + " | ".join(fields) + " |",
            "|" + "|".join(["---"] * len(fields)) + "|",
        ]
    )
    for row in combined:
        markdown.append("| " + " | ".join(str(row.get(field, "—")) for field in fields) + " |")
    markdown.append("")
    (args.out / "paper_mean_sd_tables.md").write_text("\n".join(markdown), encoding="utf-8")


if __name__ == "__main__":
    main()
