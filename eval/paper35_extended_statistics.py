"""Extended statistics and PR curve for the Paper 3.5 YOLO11n/FSDH comparison.

This script never trains or runs inference. It reuses the saved COCO prediction
JSON files from the common evaluator. Outputs are intended for the paper
appendix and its result/provenance checklist.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from paper27_bootstrap_confusion_area import (  # noqa: E402
    ap_for_area,
    bbox_iou_matrix,
    build_evaluation_cache,
    build_specs,
)

OUT = ROOT / "analysis" / "paper_3_4_review" / "paper35_extended_statistics"
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260830
CONFIDENCE = 0.25
MATCH_IOU = 0.50


def sample_sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else math.nan


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def class_aware_iou_error_per_image(
    annotations_path: Path, prediction_path: Path, image_ids: list[int]
) -> tuple[np.ndarray, np.ndarray, list[set[int]]]:
    """Return per-image sum/count of 1-IoU and matched annotation ids."""
    gt = json.loads(annotations_path.read_text(encoding="utf-8"))
    pred = json.loads(prediction_path.read_text(encoding="utf-8"))
    image_index = {image_id: idx for idx, image_id in enumerate(image_ids)}
    gt_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in image_ids}
    pred_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in image_ids}
    for item in gt["annotations"]:
        image_id = int(item["image_id"])
        if image_id in gt_by_image:
            gt_by_image[image_id].append(item)
    for item in pred:
        image_id = int(item["image_id"])
        if image_id in pred_by_image and float(item["score"]) >= CONFIDENCE:
            pred_by_image[image_id].append(item)

    sums = np.zeros(len(image_ids), dtype=np.float64)
    counts = np.zeros(len(image_ids), dtype=np.int32)
    matched_ids: list[set[int]] = [set() for _ in image_ids]
    for image_id in image_ids:
        idx = image_index[image_id]
        for category_id in sorted({int(x["category_id"]) for x in gt_by_image[image_id]}):
            ground_truth = [x for x in gt_by_image[image_id] if int(x["category_id"]) == category_id]
            detected = sorted(
                [x for x in pred_by_image[image_id] if int(x["category_id"]) == category_id],
                key=lambda x: float(x["score"]),
                reverse=True,
            )
            gt_boxes = np.asarray([x["bbox"] for x in ground_truth], dtype=np.float64).reshape(-1, 4)
            pred_boxes = np.asarray([x["bbox"] for x in detected], dtype=np.float64).reshape(-1, 4)
            ious = bbox_iou_matrix(gt_boxes, pred_boxes)
            candidates = sorted(
                ((int(g), int(p), float(ious[g, p])) for g, p in np.argwhere(ious >= MATCH_IOU)),
                key=lambda row: row[2],
                reverse=True,
            )
            used_gt: set[int] = set()
            used_pred: set[int] = set()
            for gt_idx, pred_idx, iou in candidates:
                if gt_idx in used_gt or pred_idx in used_pred:
                    continue
                used_gt.add(gt_idx)
                used_pred.add(pred_idx)
                annotation_id = int(ground_truth[gt_idx]["id"])
                matched_ids[idx].add(annotation_id)
                sums[idx] += 1.0 - iou
                counts[idx] += 1
    return sums, counts, matched_ids


def common_match_errors(
    annotations_path: Path,
    yolo_prediction: Path,
    fsdh_prediction: Path,
    image_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recompute error sums/counts over GT objects matched by both models."""
    # Build annotation-level IoU maps by repeating the matching with a compact helper.
    gt = json.loads(annotations_path.read_text(encoding="utf-8"))

    def maps(prediction_path: Path) -> list[dict[int, float]]:
        pred = json.loads(prediction_path.read_text(encoding="utf-8"))
        gt_by_image = {image_id: [] for image_id in image_ids}
        pred_by_image = {image_id: [] for image_id in image_ids}
        for item in gt["annotations"]:
            if int(item["image_id"]) in gt_by_image:
                gt_by_image[int(item["image_id"])].append(item)
        for item in pred:
            if int(item["image_id"]) in pred_by_image and float(item["score"]) >= CONFIDENCE:
                pred_by_image[int(item["image_id"])].append(item)
        result: list[dict[int, float]] = []
        for image_id in image_ids:
            matched: dict[int, float] = {}
            category_ids = sorted({int(x["category_id"]) for x in gt_by_image[image_id]})
            for category_id in category_ids:
                ground_truth = [x for x in gt_by_image[image_id] if int(x["category_id"]) == category_id]
                detected = sorted(
                    [x for x in pred_by_image[image_id] if int(x["category_id"]) == category_id],
                    key=lambda x: float(x["score"]), reverse=True,
                )
                gt_boxes = np.asarray([x["bbox"] for x in ground_truth], dtype=np.float64).reshape(-1, 4)
                pred_boxes = np.asarray([x["bbox"] for x in detected], dtype=np.float64).reshape(-1, 4)
                ious = bbox_iou_matrix(gt_boxes, pred_boxes)
                used_gt: set[int] = set()
                # Follow the fixed-workpoint evaluation convention: detections
                # are processed in descending confidence order, and each is
                # assigned to the highest-IoU still-unmatched GT of the class.
                for pred_idx in range(len(detected)):
                    ranked_gt = np.argsort(-ious[:, pred_idx]) if len(ground_truth) else []
                    for gt_idx_value in ranked_gt:
                        gt_idx = int(gt_idx_value)
                        iou = float(ious[gt_idx, pred_idx])
                        if iou < MATCH_IOU:
                            break
                        if gt_idx not in used_gt:
                            used_gt.add(gt_idx)
                            matched[int(ground_truth[gt_idx]["id"])] = iou
                            break
            result.append(matched)
        return result

    yolo_maps = maps(yolo_prediction)
    fsdh_maps = maps(fsdh_prediction)
    yolo_sums = np.zeros(len(image_ids), dtype=np.float64)
    fsdh_sums = np.zeros(len(image_ids), dtype=np.float64)
    counts = np.zeros(len(image_ids), dtype=np.int32)
    for idx, (left, right) in enumerate(zip(yolo_maps, fsdh_maps)):
        common = left.keys() & right.keys()
        counts[idx] = len(common)
        yolo_sums[idx] = sum(1.0 - left[key] for key in common)
        fsdh_sums[idx] = sum(1.0 - right[key] for key in common)
    return yolo_sums, counts.copy(), fsdh_sums, counts


def standard_pr_curve(annotations: Path, predictions: Path) -> tuple[np.ndarray, np.ndarray]:
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(annotations))
        coco_dt = coco_gt.loadRes(str(predictions))
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = sorted(coco_gt.getImgIds())
        evaluator.params.catIds = sorted(coco_gt.getCatIds())
        evaluator.params.maxDets = [1, 10, 300]
        evaluator.evaluate()
        evaluator.accumulate()
    iou_idx = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.50)))
    area_idx = evaluator.params.areaRngLbl.index("all")
    max_det_idx = evaluator.params.maxDets.index(300)
    precision = evaluator.eval["precision"][iou_idx, :, :, area_idx, max_det_idx]
    precision = np.where(precision < 0, np.nan, precision)
    return evaluator.params.recThrs.copy(), np.nanmean(precision, axis=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base_dataset = build_specs()[0]
    dataset = replace(
        base_dataset,
        area_ranges={**base_dataset.area_ranges, "coco_small": (0.0, float(32**2))},
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    annotations_payload = json.loads(dataset.annotations.read_text(encoding="utf-8"))
    image_count = len(annotations_payload["images"])
    multiplicities = rng.multinomial(
        image_count,
        np.full(image_count, 1.0 / image_count),
        size=BOOTSTRAP_REPLICATES,
    ).astype(np.int16)

    category_names: dict[int, str] = {
        int(item["id"]): str(item["name"]) for item in annotations_payload["categories"]
    }
    metric_specs: list[tuple[str, str, list[int] | None, int]] = [
        ("mAP50-95", "all", None, 0),
        ("mAP50", "all", None, 1),
        ("AP75", "all", None, 2),
        ("AP_small@640eq", "small_area4", None, 0),
        ("COCO AP_s", "coco_small", None, 0),
    ]
    for category_id, name in sorted(category_names.items()):
        metric_specs.extend(
            [
                (f"{name} AP50", "all", [category_id], 1),
                (f"{name} AP50-95", "all", [category_id], 0),
                (f"{name} AP_small@640eq", "small_area4", [category_id], 0),
                (f"{name} AP_non-small", "non_small", [category_id], 0),
            ]
        )

    seed_points: dict[str, dict[str, list[float]]] = {
        model.label: {metric[0]: [] for metric in metric_specs} for model in dataset.models
    }
    distributions: dict[str, dict[str, list[np.ndarray]]] = {
        model.label: {metric[0]: [] for metric in metric_specs} for model in dataset.models
    }
    caches: dict[str, list[Any]] = {model.label: [] for model in dataset.models}

    for model in dataset.models:
        for seed, prediction_path in enumerate(model.prediction_files):
            print(f"Caching {model.label} seed {seed}", flush=True)
            cache = build_evaluation_cache(dataset, prediction_path)
            caches[model.label].append(cache)
            full = np.ones((1, image_count), dtype=np.int16)
            for metric_name, area, category_ids, output_idx in metric_specs:
                point_tuple = ap_for_area(cache, full, area, category_ids)
                bootstrap_tuple = ap_for_area(cache, multiplicities, area, category_ids)
                seed_points[model.label][metric_name].append(float(point_tuple[output_idx][0]))
                distributions[model.label][metric_name].append(bootstrap_tuple[output_idx])

    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {
        "schema": "paper35_extended_statistics_v1",
        "protocol": {
            "test_images": image_count,
            "bootstrap_unit": "test image",
            "bootstrap_type": "paired nonparametric percentile bootstrap",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "aggregation": "same bootstrap image multiplicities for both models and all three seeds; mean across seeds",
            "COCO_maxDet": 300,
            "IoU_error_confidence": CONFIDENCE,
            "IoU_error_match_IoU": MATCH_IOU,
        },
        "annotations": str(dataset.annotations.resolve()),
        "inputs": {
            model.label: [str(path.resolve()) for path in model.prediction_files]
            for model in dataset.models
        },
        "metrics": {},
    }
    for metric_name, *_ in metric_specs:
        yolo_seed = seed_points["YOLO11n"][metric_name]
        fsdh_seed = seed_points["FSDH-YOLO"][metric_name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            yolo_dist = np.nanmean(np.stack(distributions["YOLO11n"][metric_name]), axis=0)
            fsdh_dist = np.nanmean(np.stack(distributions["FSDH-YOLO"][metric_name]), axis=0)
        delta_dist = fsdh_dist - yolo_dist
        finite_delta = delta_dist[np.isfinite(delta_dist)]
        delta_seed = [right - left for left, right in zip(yolo_seed, fsdh_seed)]
        item = {
            "YOLO11n_per_seed": yolo_seed,
            "YOLO11n_mean": float(np.mean(yolo_seed)),
            "YOLO11n_sample_sd": sample_sd(yolo_seed),
            "FSDH_per_seed": fsdh_seed,
            "FSDH_mean": float(np.mean(fsdh_seed)),
            "FSDH_sample_sd": sample_sd(fsdh_seed),
            "delta_per_seed": delta_seed,
            "delta_mean": float(np.mean(delta_seed)),
            "delta_sample_sd": sample_sd(delta_seed),
            "bootstrap_ci": [float(np.nanpercentile(delta_dist, 2.5)), float(np.nanpercentile(delta_dist, 97.5))],
            "bootstrap_defined_replicates": int(finite_delta.size),
            "bootstrap_positive_fraction": float(np.mean(finite_delta > 0)),
        }
        results["metrics"][metric_name] = item
        rows.append(
            {
                "metric": metric_name,
                "YOLO11n_mean_percent": item["YOLO11n_mean"] * 100,
                "YOLO11n_sd_percent": item["YOLO11n_sample_sd"] * 100,
                "FSDH_mean_percent": item["FSDH_mean"] * 100,
                "FSDH_sd_percent": item["FSDH_sample_sd"] * 100,
                "delta_percent_points": item["delta_mean"] * 100,
                "delta_sd_percent_points": item["delta_sample_sd"] * 100,
                "bootstrap_ci_lower_percent_points": item["bootstrap_ci"][0] * 100,
                "bootstrap_ci_upper_percent_points": item["bootstrap_ci"][1] * 100,
                "bootstrap_positive_fraction": item["bootstrap_positive_fraction"],
            }
        )

    # IoU localization error on common matched GT objects.
    yolo_error_seed: list[float] = []
    fsdh_error_seed: list[float] = []
    yolo_error_dist_seed: list[np.ndarray] = []
    fsdh_error_dist_seed: list[np.ndarray] = []
    common_counts: list[int] = []
    image_ids = caches["YOLO11n"][0].image_ids
    for seed in range(3):
        ys, yc, fs, fc = common_match_errors(
            dataset.annotations,
            dataset.models[0].prediction_files[seed],
            dataset.models[1].prediction_files[seed],
            image_ids,
        )
        common_counts.append(int(yc.sum()))
        yolo_error_seed.append(float(ys.sum() / yc.sum()))
        fsdh_error_seed.append(float(fs.sum() / fc.sum()))
        weighted_count = multiplicities @ yc.astype(np.float64)
        yolo_error_dist_seed.append((multiplicities @ ys) / weighted_count)
        fsdh_error_dist_seed.append((multiplicities @ fs) / weighted_count)
    yolo_error_dist = np.mean(np.stack(yolo_error_dist_seed), axis=0)
    fsdh_error_dist = np.mean(np.stack(fsdh_error_dist_seed), axis=0)
    delta_error_dist = fsdh_error_dist - yolo_error_dist
    delta_error_seed = [right - left for left, right in zip(yolo_error_seed, fsdh_error_seed)]
    iou_item = {
        "common_matches_per_seed": common_counts,
        "YOLO11n_per_seed": yolo_error_seed,
        "YOLO11n_mean": float(np.mean(yolo_error_seed)),
        "YOLO11n_sample_sd": sample_sd(yolo_error_seed),
        "FSDH_per_seed": fsdh_error_seed,
        "FSDH_mean": float(np.mean(fsdh_error_seed)),
        "FSDH_sample_sd": sample_sd(fsdh_error_seed),
        "delta_per_seed": delta_error_seed,
        "delta_mean": float(np.mean(delta_error_seed)),
        "delta_sample_sd": sample_sd(delta_error_seed),
        "bootstrap_ci": [float(np.percentile(delta_error_dist, 2.5)), float(np.percentile(delta_error_dist, 97.5))],
        "bootstrap_negative_fraction": float(np.mean(delta_error_dist < 0)),
    }
    results["metrics"]["IoU localization error"] = iou_item
    rows.append(
        {
            "metric": "IoU localization error",
            "YOLO11n_mean_percent": iou_item["YOLO11n_mean"] * 100,
            "YOLO11n_sd_percent": iou_item["YOLO11n_sample_sd"] * 100,
            "FSDH_mean_percent": iou_item["FSDH_mean"] * 100,
            "FSDH_sd_percent": iou_item["FSDH_sample_sd"] * 100,
            "delta_percent_points": iou_item["delta_mean"] * 100,
            "delta_sd_percent_points": iou_item["delta_sample_sd"] * 100,
            "bootstrap_ci_lower_percent_points": iou_item["bootstrap_ci"][0] * 100,
            "bootstrap_ci_upper_percent_points": iou_item["bootstrap_ci"][1] * 100,
            "bootstrap_positive_fraction": 1.0 - iou_item["bootstrap_negative_fraction"],
        }
    )

    # Three-seed COCO interpolated PR curve at IoU=0.50, averaged across classes.
    recall: np.ndarray | None = None
    curves: dict[str, list[np.ndarray]] = {model.label: [] for model in dataset.models}
    pr_rows: list[dict[str, Any]] = []
    for model in dataset.models:
        for seed, prediction_path in enumerate(model.prediction_files):
            recall, precision = standard_pr_curve(dataset.annotations, prediction_path)
            curves[model.label].append(precision)
            for r_value, p_value in zip(recall, precision):
                pr_rows.append({"model": model.label, "seed": seed, "recall": r_value, "precision": p_value})

    colors = {"YOLO11n": "#4C78A8", "FSDH-YOLO": "#E45756"}
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=220)
    assert recall is not None
    for label in ["YOLO11n", "FSDH-YOLO"]:
        stacked = np.stack(curves[label])
        mean = np.nanmean(stacked, axis=0)
        sd = np.nanstd(stacked, axis=0, ddof=1)
        ap50 = results["metrics"]["mAP50"]
        metric_mean = ap50["YOLO11n_mean"] if label == "YOLO11n" else ap50["FSDH_mean"]
        metric_sd = ap50["YOLO11n_sample_sd"] if label == "YOLO11n" else ap50["FSDH_sample_sd"]
        ax.plot(recall, mean, color=colors[label], linewidth=2.2, label=f"{label} (AP50={metric_mean*100:.2f}±{metric_sd*100:.2f}%)")
        ax.fill_between(recall, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), color=colors[label], alpha=0.16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Interpolated precision")
    ax.set_title("Three-seed mean P–R curve at IoU = 0.50")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "YOLO11n_vs_FSDH_PR50_three_seed.png", bbox_inches="tight")
    fig.savefig(OUT / "YOLO11n_vs_FSDH_PR50_three_seed.pdf", bbox_inches="tight")
    plt.close(fig)

    write_csv(OUT / "bootstrap_ci_extended.csv", rows)
    write_csv(OUT / "pr50_curve_data.csv", pr_rows)
    (OUT / "bootstrap_results_extended.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote outputs to {OUT}", flush=True)


if __name__ == "__main__":
    main()
