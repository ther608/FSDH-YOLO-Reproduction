"""Paper 2.7 statistical supplement from saved test-set COCO predictions.

This script does not train or run inference. It reuses the three-seed test
predictions already exported by the common evaluator and produces:
1) image-level paired percentile-bootstrap 95% CIs;
2) three-seed mean row-normalized detection confusion matrices;
3) Dataset-A and Dataset-B class-by-area AP tables, plus Dataset-B AP_s/m/l.

The bootstrap is exact with respect to COCOeval's per-image matching: sampled
image multiplicities are used as weights when accumulating cached per-image
COCO evaluation records. AP is recomputed for every bootstrap sample; per-image
AP values are never averaged.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "analysis" / "paper27_statistical_supplement"
IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)
RECALL_THRESHOLDS = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    prediction_files: tuple[Path, Path, Path]
    metric_files: tuple[Path, Path, Path]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    annotations: Path
    area_ranges: dict[str, tuple[float, float]]
    bootstrap_small_label: str
    models: tuple[ModelSpec, ModelSpec]


@dataclass
class ClassAreaCache:
    detection_image_indices: np.ndarray
    true_positive: np.ndarray
    false_positive: np.ndarray
    nonignored_gt_per_image: np.ndarray


@dataclass
class EvaluationCache:
    image_ids: list[int]
    category_ids: list[int]
    category_names: dict[int, str]
    area_ranges: dict[str, tuple[float, float]]
    entries: dict[tuple[str, int], ClassAreaCache]


def build_specs() -> tuple[DatasetSpec, DatasetSpec]:
    common_a = ROOT / "runs" / "common_eval" / "paper16_unified_protocol"
    common_b = ROOT / "runs" / "common_eval" / "paper_dataset_b_groupaware_clean_v4_aug_comparison_v1"

    def files(directory: Path, kind: str) -> tuple[Path, Path, Path]:
        return tuple(directory / f"seed{seed}__test_{kind}_common.json" for seed in range(3))  # type: ignore[return-value]

    def predictions(directory: Path) -> tuple[Path, Path, Path]:
        return tuple(directory / f"seed{seed}__test_predictions_coco.json" for seed in range(3))  # type: ignore[return-value]

    dataset_a = DatasetSpec(
        key="dataset_a",
        label="Dataset A",
        annotations=ROOT / "external_configs" / "rtdetr_coco_annotations" / "instances_test.json",
        area_ranges={
            "all": (0.0, 1.0e10),
            "small_area4": (0.0, float(32**2 * 4)),
            "non_small": (float(32**2 * 4), 1.0e10),
        },
        bootstrap_small_label="small_area4",
        models=(
            ModelSpec(
                "YOLO11n",
                predictions(common_a / "yolo11n"),
                files(common_a / "yolo11n", "metrics"),
            ),
            ModelSpec(
                "FSDH-YOLO",
                predictions(common_a / "fsdconv_shab"),
                files(common_a / "fsdconv_shab", "metrics"),
            ),
        ),
    )
    dataset_b = DatasetSpec(
        key="dataset_b",
        label="Dataset B",
        annotations=ROOT / "external_configs" / "dataset_b_native4_v4" / "instances_test.json",
        area_ranges={
            "all": (0.0, 1.0e10),
            "small": (0.0, float(32**2)),
            "non_small": (float(32**2), 1.0e10),
            "medium": (float(32**2), float(96**2)),
            "large": (float(96**2), 1.0e10),
        },
        bootstrap_small_label="small",
        models=(
            ModelSpec(
                "YOLO11n",
                predictions(common_b / "DatasetB4_YOLO11n"),
                files(common_b / "DatasetB4_YOLO11n", "metrics"),
            ),
            ModelSpec(
                "FSDH-YOLO",
                predictions(common_b / "DatasetB4_FSDH_YOLO"),
                files(common_b / "DatasetB4_FSDH_YOLO", "metrics"),
            ),
        ),
    )
    return dataset_a, dataset_b


def ensure_inputs(specs: tuple[DatasetSpec, ...]) -> None:
    missing: list[str] = []
    for dataset in specs:
        for path in [dataset.annotations]:
            if not path.is_file():
                missing.append(str(path))
        for model in dataset.models:
            for path in (*model.prediction_files, *model.metric_files):
                if not path.is_file():
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def build_evaluation_cache(dataset: DatasetSpec, prediction_path: Path) -> EvaluationCache:
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(dataset.annotations))
        coco_dt = coco_gt.loadRes(str(prediction_path))
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = sorted(coco_gt.getImgIds())
        evaluator.params.catIds = sorted(coco_gt.getCatIds())
        evaluator.params.maxDets = [1, 10, 300]
        evaluator.params.iouThrs = IOU_THRESHOLDS
        evaluator.params.areaRng = [list(dataset.area_ranges[name]) for name in dataset.area_ranges]
        evaluator.params.areaRngLbl = list(dataset.area_ranges)
        evaluator.evaluate()

    image_ids = list(evaluator.params.imgIds)
    category_ids = list(evaluator.params.catIds)
    image_to_index = {image_id: index for index, image_id in enumerate(image_ids)}
    category_names = {item["id"]: item["name"] for item in coco_gt.loadCats(category_ids)}
    area_lookup = {
        tuple(float(value) for value in values): label for label, values in dataset.area_ranges.items()
    }
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {
        (area_label, category_id): []
        for area_label in dataset.area_ranges
        for category_id in category_ids
    }
    for item in evaluator.evalImgs:
        if item is None:
            continue
        area_label = area_lookup[tuple(float(value) for value in item["aRng"])]
        grouped[(area_label, int(item["category_id"]))].append(item)

    entries: dict[tuple[str, int], ClassAreaCache] = {}
    for key, eval_images in grouped.items():
        scores: list[np.ndarray] = []
        detection_image_indices: list[np.ndarray] = []
        true_positive: list[np.ndarray] = []
        false_positive: list[np.ndarray] = []
        gt_per_image = np.zeros(len(image_ids), dtype=np.int32)
        for item in eval_images:
            image_index = image_to_index[int(item["image_id"])]
            gt_ignore = np.asarray(item["gtIgnore"], dtype=bool)
            gt_per_image[image_index] = int((~gt_ignore).sum())
            dt_scores = np.asarray(item["dtScores"], dtype=np.float64)
            dt_matches = np.asarray(item["dtMatches"], dtype=np.float64)
            dt_ignore = np.asarray(item["dtIgnore"], dtype=bool)
            scores.append(dt_scores)
            detection_image_indices.append(np.full(dt_scores.size, image_index, dtype=np.int32))
            true_positive.append((dt_matches > 0) & (~dt_ignore))
            false_positive.append((dt_matches == 0) & (~dt_ignore))
        if scores:
            all_scores = np.concatenate(scores)
            order = np.argsort(-all_scores, kind="mergesort")
            all_image_indices = np.concatenate(detection_image_indices)[order]
            all_tp = np.concatenate(true_positive, axis=1)[:, order]
            all_fp = np.concatenate(false_positive, axis=1)[:, order]
        else:
            all_image_indices = np.empty(0, dtype=np.int32)
            all_tp = np.empty((len(IOU_THRESHOLDS), 0), dtype=bool)
            all_fp = np.empty((len(IOU_THRESHOLDS), 0), dtype=bool)
        entries[key] = ClassAreaCache(all_image_indices, all_tp, all_fp, gt_per_image)
    return EvaluationCache(image_ids, category_ids, category_names, dataset.area_ranges, entries)


def ap_for_area(
    cache: EvaluationCache,
    image_multiplicities: np.ndarray,
    area_label: str,
    category_ids: list[int] | None = None,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return COCO AP50-95, AP50, and AP75 for each image-bootstrap sample."""
    if category_ids is None:
        category_ids = cache.category_ids
    samples = image_multiplicities.shape[0]
    map_values = np.full(samples, np.nan, dtype=np.float64)
    ap50_values = np.full(samples, np.nan, dtype=np.float64)
    ap75_values = np.full(samples, np.nan, dtype=np.float64)
    recall_thresholds = RECALL_THRESHOLDS

    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        counts = torch.as_tensor(image_multiplicities[start:stop], dtype=torch.float64)
        per_class_all: list[torch.Tensor] = []
        per_class_50: list[torch.Tensor] = []
        per_class_75: list[torch.Tensor] = []
        for category_id in category_ids:
            item = cache.entries[(area_label, category_id)]
            gt_counts = torch.as_tensor(item.nonignored_gt_per_image, dtype=torch.float64)
            nonignored_gt = counts @ gt_counts
            if item.detection_image_indices.size == 0:
                class_all = torch.zeros(stop - start, dtype=torch.float64)
                class_50 = torch.zeros(stop - start, dtype=torch.float64)
                class_all[nonignored_gt <= 0] = torch.nan
                class_50[nonignored_gt <= 0] = torch.nan
                per_class_all.append(class_all)
                per_class_50.append(class_50)
                per_class_75.append(class_50.clone())
                continue

            image_indices = torch.as_tensor(item.detection_image_indices, dtype=torch.long)
            detection_weights = counts[:, image_indices]
            tp_flags = torch.as_tensor(item.true_positive, dtype=torch.float64)
            fp_flags = torch.as_tensor(item.false_positive, dtype=torch.float64)
            threshold_aps: list[torch.Tensor] = []
            for threshold_index in range(tp_flags.shape[0]):
                tp = torch.cumsum(detection_weights * tp_flags[threshold_index], dim=1)
                fp = torch.cumsum(detection_weights * fp_flags[threshold_index], dim=1)
                recall = tp / torch.clamp(nonignored_gt[:, None], min=1.0)
                precision = tp / torch.clamp(tp + fp, min=torch.finfo(torch.float64).eps)
                precision = torch.flip(torch.cummax(torch.flip(precision, dims=[1]), dim=1).values, dims=[1])
                targets = recall_thresholds[None, :].expand(stop - start, -1)
                indices = torch.searchsorted(recall.contiguous(), targets.contiguous(), right=False)
                valid = indices < precision.shape[1]
                safe_indices = torch.clamp(indices, max=max(precision.shape[1] - 1, 0))
                sampled_precision = torch.gather(precision, 1, safe_indices)
                sampled_precision[~valid] = 0.0
                ap = sampled_precision.mean(dim=1)
                ap[nonignored_gt <= 0] = torch.nan
                threshold_aps.append(ap)
            stacked = torch.stack(threshold_aps, dim=1)
            per_class_all.append(torch.nanmean(stacked, dim=1))
            per_class_50.append(stacked[:, 0])
            per_class_75.append(stacked[:, 5])
        map_values[start:stop] = torch.nanmean(torch.stack(per_class_all, dim=1), dim=1).numpy()
        ap50_values[start:stop] = torch.nanmean(torch.stack(per_class_50, dim=1), dim=1).numpy()
        ap75_values[start:stop] = torch.nanmean(torch.stack(per_class_75, dim=1), dim=1).numpy()
    return map_values, ap50_values, ap75_values


def expected_metrics(metric_path: Path, small_key: str) -> dict[str, float]:
    payload = load_json(metric_path)
    overall = payload["overall"]
    return {
        "mAP50-95": float(overall["map50_95"]),
        "mAP50": float(overall["map50"]),
        "AP75": float(overall["ap75"]),
        "AP_small": float(overall[small_key]),
    }


def point_metrics(cache: EvaluationCache, dataset: DatasetSpec) -> dict[str, float]:
    full = np.ones((1, len(cache.image_ids)), dtype=np.int16)
    map_all, ap50, ap75 = ap_for_area(cache, full, "all")
    small, _, _ = ap_for_area(cache, full, dataset.bootstrap_small_label)
    return {
        "mAP50-95": float(map_all[0]),
        "mAP50": float(ap50[0]),
        "AP75": float(ap75[0]),
        "AP_small": float(small[0]),
    }


def ci_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "lower": float(np.nanpercentile(values, 2.5)),
        "upper": float(np.nanpercentile(values, 97.5)),
    }


def sample_sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else math.nan


def bbox_iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    if gt_boxes.size == 0 or pred_boxes.size == 0:
        return np.zeros((gt_boxes.shape[0], pred_boxes.shape[0]), dtype=np.float64)
    gx1, gy1, gw, gh = gt_boxes.T
    px1, py1, pw, ph = pred_boxes.T
    gx2, gy2 = gx1 + gw, gy1 + gh
    px2, py2 = px1 + pw, py1 + ph
    ix1 = np.maximum(gx1[:, None], px1[None, :])
    iy1 = np.maximum(gy1[:, None], py1[None, :])
    ix2 = np.minimum(gx2[:, None], px2[None, :])
    iy2 = np.minimum(gy2[:, None], py2[None, :])
    intersection = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = gw[:, None] * gh[:, None] + pw[None, :] * ph[None, :] - intersection
    return intersection / np.maximum(union, np.finfo(np.float64).eps)


def confusion_counts(
    annotation_path: Path,
    prediction_path: Path,
    confidence: float,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    gt = load_json(annotation_path)
    predictions = load_json(prediction_path)
    category_ids = sorted(int(item["id"]) for item in gt["categories"])
    category_index = {category_id: index for index, category_id in enumerate(category_ids)}
    category_names = [
        next(item["name"] for item in gt["categories"] if int(item["id"]) == category_id)
        for category_id in category_ids
    ]
    images = [int(item["id"]) for item in gt["images"]]
    gt_by_image = {image_id: [] for image_id in images}
    pred_by_image = {image_id: [] for image_id in images}
    for item in gt["annotations"]:
        gt_by_image[int(item["image_id"])].append(item)
    for item in predictions:
        if float(item["score"]) >= confidence:
            pred_by_image[int(item["image_id"])].append(item)

    matrix = np.zeros((len(category_ids), len(category_ids) + 1), dtype=np.int64)
    false_positives = np.zeros(len(category_ids), dtype=np.int64)
    for image_id in images:
        ground_truth = gt_by_image[image_id]
        detected = sorted(pred_by_image[image_id], key=lambda item: float(item["score"]), reverse=True)
        gt_boxes = np.asarray([item["bbox"] for item in ground_truth], dtype=np.float64).reshape(-1, 4)
        pred_boxes = np.asarray([item["bbox"] for item in detected], dtype=np.float64).reshape(-1, 4)
        ious = bbox_iou_matrix(gt_boxes, pred_boxes)
        candidates = np.argwhere(ious >= iou_threshold)
        matches: list[tuple[int, int]] = []
        if candidates.size:
            ranked = sorted(
                ((int(g), int(p), float(ious[g, p])) for g, p in candidates),
                key=lambda row: row[2],
                reverse=True,
            )
            used_gt: set[int] = set()
            used_pred: set[int] = set()
            for gt_index, pred_index, _ in ranked:
                if gt_index not in used_gt and pred_index not in used_pred:
                    matches.append((gt_index, pred_index))
                    used_gt.add(gt_index)
                    used_pred.add(pred_index)
        else:
            used_gt, used_pred = set(), set()

        for gt_index, pred_index in matches:
            true_category = category_index[int(ground_truth[gt_index]["category_id"])]
            predicted_category = category_index[int(detected[pred_index]["category_id"])]
            matrix[true_category, predicted_category] += 1
        for gt_index, item in enumerate(ground_truth):
            if gt_index not in used_gt:
                matrix[category_index[int(item["category_id"])], -1] += 1
        for pred_index, item in enumerate(detected):
            if pred_index not in used_pred:
                false_positives[category_index[int(item["category_id"])]] += 1
    return matrix, false_positives, category_ids, category_names


def plot_confusion(
    means: np.ndarray,
    names: list[str],
    title: str,
    path: Path,
) -> None:
    fig_width = max(7.0, 1.25 * (len(names) + 1))
    fig_height = max(5.0, 1.05 * len(names) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    image = ax.imshow(means * 100.0, cmap="Blues", vmin=0.0, vmax=100.0)
    ax.set_xticks(range(len(names) + 1), labels=[*names, "Miss/FN"], rotation=25, ha="right")
    ax.set_yticks(range(len(names)), labels=names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for row in range(means.shape[0]):
        for column in range(means.shape[1]):
            value = means[row, column] * 100.0
            ax.text(column, row, f"{value:.1f}", ha="center", va="center", color="white" if value > 50 else "black", fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Row-normalized proportion (%)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def gt_count_by_area(cache: EvaluationCache, area_label: str, category_id: int | None = None) -> int:
    category_ids = cache.category_ids if category_id is None else [category_id]
    return int(sum(cache.entries[(area_label, cid)].nonignored_gt_per_image.sum() for cid in category_ids))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--confusion-iou", type=float, default=0.50)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1000:
        raise ValueError("Use at least 1000 bootstrap replicates")
    args.out.mkdir(parents=True, exist_ok=True)
    specs = build_specs()
    ensure_inputs(specs)
    rng = np.random.default_rng(args.bootstrap_seed)

    results: dict[str, Any] = {
        "schema": "paper27_statistical_supplement_v2",
        "protocol": {
            "bootstrap_unit": "test image",
            "bootstrap_type": "paired nonparametric percentile bootstrap",
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "seed_aggregation": "same-seed paired difference, then arithmetic mean across seeds 0/1/2",
            "confidence_interval": "2.5th and 97.5th percentiles",
            "coco_iou_thresholds": IOU_THRESHOLDS.tolist(),
            "coco_recall_thresholds": 101,
            "coco_max_det": 300,
            "confusion_confidence": args.conf,
            "confusion_iou": args.confusion_iou,
            "confusion_normalization": "each true-class row sums to one, including Miss/FN",
        },
        "datasets": {},
    }
    bootstrap_csv: list[dict[str, Any]] = []
    area_a_csv: list[dict[str, Any]] = []
    area_b_csv: list[dict[str, Any]] = []
    area_b_class_csv: list[dict[str, Any]] = []
    confusion_csv: list[dict[str, Any]] = []
    distributions: dict[str, np.ndarray] = {}

    for dataset in specs:
        print(f"Preparing {dataset.label} ({dataset.annotations})", flush=True)
        annotation_payload = load_json(dataset.annotations)
        image_count = len(annotation_payload["images"])
        annotation_count = len(annotation_payload["annotations"])
        multiplicities = rng.multinomial(
            image_count,
            np.full(image_count, 1.0 / image_count),
            size=args.bootstrap_replicates,
        ).astype(np.int16)
        dataset_result: dict[str, Any] = {
            "annotations": str(dataset.annotations.resolve()),
            "test_images": image_count,
            "evaluation_annotations": annotation_count,
            "area_ranges": {key: list(value) for key, value in dataset.area_ranges.items()},
            "models": {},
        }
        model_bootstrap: dict[str, dict[str, np.ndarray]] = {}
        model_caches: dict[str, list[EvaluationCache]] = {}

        for model in dataset.models:
            print(f"  {model.label}: caching and bootstrapping 3 seeds", flush=True)
            model_result: dict[str, Any] = {"seeds": [], "bootstrap": {}}
            seed_distributions: dict[str, list[np.ndarray]] = {
                "mAP50-95": [],
                "mAP50": [],
                "AP75": [],
                "AP_small": [],
            }
            caches: list[EvaluationCache] = []
            for seed, (prediction_path, metric_path) in enumerate(zip(model.prediction_files, model.metric_files)):
                print(f"    seed {seed}", flush=True)
                cache = build_evaluation_cache(dataset, prediction_path)
                caches.append(cache)
                point = point_metrics(cache, dataset)
                small_key = "ap_small_area4" if dataset.key == "dataset_a" else "ap_small"
                expected = expected_metrics(metric_path, small_key)
                differences = {metric: abs(point[metric] - expected[metric]) for metric in point}
                if max(differences.values()) > 1e-10:
                    raise AssertionError(
                        f"Point metric mismatch for {dataset.label}/{model.label}/seed{seed}: {differences}"
                    )
                all_map, all_ap50, all_ap75 = ap_for_area(cache, multiplicities, "all", batch_size=args.batch_size)
                small_map, _, _ = ap_for_area(
                    cache,
                    multiplicities,
                    dataset.bootstrap_small_label,
                    batch_size=args.batch_size,
                )
                seed_distributions["mAP50-95"].append(all_map)
                seed_distributions["mAP50"].append(all_ap50)
                seed_distributions["AP75"].append(all_ap75)
                seed_distributions["AP_small"].append(small_map)
                model_result["seeds"].append(
                    {
                        "seed": seed,
                        "predictions": str(prediction_path.resolve()),
                        "reference_metrics": str(metric_path.resolve()),
                        "point": point,
                        "point_validation_abs_error": differences,
                    }
                )
            model_caches[model.label] = caches
            model_bootstrap[model.label] = {}
            for metric, seed_arrays in seed_distributions.items():
                average_distribution = np.mean(np.stack(seed_arrays, axis=0), axis=0)
                model_bootstrap[model.label][metric] = average_distribution
                seed_points = [row["point"][metric] for row in model_result["seeds"]]
                summary = {
                    "point_mean": float(np.mean(seed_points)),
                    "seed_sample_sd": sample_sd(seed_points),
                    "bootstrap_95ci": ci_summary(average_distribution),
                }
                model_result["bootstrap"][metric] = summary
                distributions[f"{dataset.key}__{model.label}__{metric}"] = average_distribution
                bootstrap_csv.append(
                    {
                        "dataset": dataset.label,
                        "comparison": model.label,
                        "metric": metric,
                        "point_percent": summary["point_mean"] * 100.0,
                        "seed_sample_sd_percent": summary["seed_sample_sd"] * 100.0,
                        "bootstrap_ci_lower_percent": summary["bootstrap_95ci"]["lower"] * 100.0,
                        "bootstrap_ci_upper_percent": summary["bootstrap_95ci"]["upper"] * 100.0,
                    }
                )
            dataset_result["models"][model.label] = model_result

        paired: dict[str, Any] = {}
        for metric in ["mAP50-95", "mAP50", "AP75", "AP_small"]:
            delta_distribution = model_bootstrap["FSDH-YOLO"][metric] - model_bootstrap["YOLO11n"][metric]
            yolo_points = [row["point"][metric] for row in dataset_result["models"]["YOLO11n"]["seeds"]]
            fsdh_points = [row["point"][metric] for row in dataset_result["models"]["FSDH-YOLO"]["seeds"]]
            paired_seed_differences = [right - left for left, right in zip(yolo_points, fsdh_points)]
            summary = {
                "point_mean_difference": float(np.mean(paired_seed_differences)),
                "paired_seed_difference_sample_sd": sample_sd(paired_seed_differences),
                "bootstrap_95ci": ci_summary(delta_distribution),
                "bootstrap_positive_fraction": float(np.mean(delta_distribution > 0)),
                "bootstrap_negative_fraction": float(np.mean(delta_distribution < 0)),
            }
            paired[metric] = summary
            distributions[f"{dataset.key}__FSDH_minus_YOLO11n__{metric}"] = delta_distribution
            bootstrap_csv.append(
                {
                    "dataset": dataset.label,
                    "comparison": "FSDH-YOLO minus YOLO11n",
                    "metric": metric,
                    "point_percent": summary["point_mean_difference"] * 100.0,
                    "seed_sample_sd_percent": summary["paired_seed_difference_sample_sd"] * 100.0,
                    "bootstrap_ci_lower_percent": summary["bootstrap_95ci"]["lower"] * 100.0,
                    "bootstrap_ci_upper_percent": summary["bootstrap_95ci"]["upper"] * 100.0,
                }
            )
        dataset_result["paired_fsdh_minus_yolo11n"] = paired

        print(f"  {dataset.label}: area-stratified AP", flush=True)
        if dataset.key == "dataset_a":
            area_result: list[dict[str, Any]] = []
            reference_cache = model_caches["YOLO11n"][0]
            targets: list[tuple[str, int | None]] = [("All", None)] + [
                (reference_cache.category_names[cid], cid) for cid in reference_cache.category_ids
            ]
            for target_name, category_id in targets:
                for area_label in ["small_area4", "non_small"]:
                    row: dict[str, Any] = {
                        "target": target_name,
                        "category_id": category_id,
                        "area": area_label,
                        "gt_count": gt_count_by_area(reference_cache, area_label, category_id),
                        "models": {},
                    }
                    for model in dataset.models:
                        seed_values: list[float] = []
                        for cache in model_caches[model.label]:
                            counts = np.ones((1, len(cache.image_ids)), dtype=np.int16)
                            values, _, _ = ap_for_area(
                                cache,
                                counts,
                                area_label,
                                None if category_id is None else [category_id],
                            )
                            seed_values.append(float(values[0]))
                        row["models"][model.label] = {
                            "per_seed": seed_values,
                            "mean": float(np.mean(seed_values)),
                            "sample_sd": sample_sd(seed_values),
                        }
                    row["mean_difference"] = (
                        row["models"]["FSDH-YOLO"]["mean"] - row["models"]["YOLO11n"]["mean"]
                    )
                    area_result.append(row)
                    area_a_csv.append(
                        {
                            "target": target_name,
                            "category_id": "all" if category_id is None else category_id,
                            "area": area_label,
                            "gt_count": row["gt_count"],
                            "YOLO11n_mean_percent": row["models"]["YOLO11n"]["mean"] * 100.0,
                            "YOLO11n_sample_sd_percent": row["models"]["YOLO11n"]["sample_sd"] * 100.0,
                            "FSDH_mean_percent": row["models"]["FSDH-YOLO"]["mean"] * 100.0,
                            "FSDH_sample_sd_percent": row["models"]["FSDH-YOLO"]["sample_sd"] * 100.0,
                            "difference_percent_points": row["mean_difference"] * 100.0,
                        }
                    )
            dataset_result["area_ap"] = area_result
        else:
            area_result = []
            reference_cache = model_caches["YOLO11n"][0]
            for area_label in ["small", "medium", "large"]:
                row = {
                    "area": area_label,
                    "gt_count": gt_count_by_area(reference_cache, area_label),
                    "models": {},
                }
                for model in dataset.models:
                    seed_values = []
                    for cache in model_caches[model.label]:
                        counts = np.ones((1, len(cache.image_ids)), dtype=np.int16)
                        values, _, _ = ap_for_area(cache, counts, area_label)
                        seed_values.append(float(values[0]))
                    row["models"][model.label] = {
                        "per_seed": seed_values,
                        "mean": float(np.mean(seed_values)),
                        "sample_sd": sample_sd(seed_values),
                    }
                row["mean_difference"] = row["models"]["FSDH-YOLO"]["mean"] - row["models"]["YOLO11n"]["mean"]
                area_result.append(row)
                area_b_csv.append(
                    {
                        "area": area_label,
                        "gt_count": row["gt_count"],
                        "YOLO11n_mean_percent": row["models"]["YOLO11n"]["mean"] * 100.0,
                        "YOLO11n_sample_sd_percent": row["models"]["YOLO11n"]["sample_sd"] * 100.0,
                        "FSDH_mean_percent": row["models"]["FSDH-YOLO"]["mean"] * 100.0,
                        "FSDH_sample_sd_percent": row["models"]["FSDH-YOLO"]["sample_sd"] * 100.0,
                        "difference_percent_points": row["mean_difference"] * 100.0,
                    }
                )
            dataset_result["area_ap"] = area_result

            class_area_result: list[dict[str, Any]] = []
            targets = [("All", None)] + [
                (reference_cache.category_names[cid], cid) for cid in reference_cache.category_ids
            ]
            for target_name, category_id in targets:
                for area_label in ["small", "non_small"]:
                    row = {
                        "target": target_name,
                        "category_id": category_id,
                        "area": area_label,
                        "gt_count": gt_count_by_area(reference_cache, area_label, category_id),
                        "models": {},
                    }
                    for model in dataset.models:
                        seed_values = []
                        for cache in model_caches[model.label]:
                            counts = np.ones((1, len(cache.image_ids)), dtype=np.int16)
                            values, _, _ = ap_for_area(
                                cache,
                                counts,
                                area_label,
                                None if category_id is None else [category_id],
                            )
                            seed_values.append(float(values[0]))
                        row["models"][model.label] = {
                            "per_seed": seed_values,
                            "mean": float(np.mean(seed_values)),
                            "sample_sd": sample_sd(seed_values),
                        }
                    row["mean_difference"] = (
                        row["models"]["FSDH-YOLO"]["mean"] - row["models"]["YOLO11n"]["mean"]
                    )
                    class_area_result.append(row)
                    area_b_class_csv.append(
                        {
                            "target": target_name,
                            "category_id": "all" if category_id is None else category_id,
                            "area": area_label,
                            "gt_count": row["gt_count"],
                            "YOLO11n_mean_percent": row["models"]["YOLO11n"]["mean"] * 100.0,
                            "YOLO11n_sample_sd_percent": row["models"]["YOLO11n"]["sample_sd"] * 100.0,
                            "FSDH_mean_percent": row["models"]["FSDH-YOLO"]["mean"] * 100.0,
                            "FSDH_sample_sd_percent": row["models"]["FSDH-YOLO"]["sample_sd"] * 100.0,
                            "difference_percent_points": row["mean_difference"] * 100.0,
                        }
                    )
            dataset_result["class_area_ap"] = class_area_result

        print(f"  {dataset.label}: confusion matrices", flush=True)
        confusion_result: dict[str, Any] = {}
        for model in dataset.models:
            matrices: list[np.ndarray] = []
            normalized: list[np.ndarray] = []
            false_positives: list[np.ndarray] = []
            class_names: list[str] | None = None
            for seed, prediction_path in enumerate(model.prediction_files):
                matrix, fp, _, names = confusion_counts(
                    dataset.annotations,
                    prediction_path,
                    args.conf,
                    args.confusion_iou,
                )
                denominators = matrix.sum(axis=1, keepdims=True)
                row_normalized = matrix / np.maximum(denominators, 1)
                matrices.append(matrix)
                normalized.append(row_normalized)
                false_positives.append(fp)
                class_names = names
            normalized_stack = np.stack(normalized, axis=0)
            matrix_stack = np.stack(matrices, axis=0)
            fp_stack = np.stack(false_positives, axis=0)
            mean_normalized = normalized_stack.mean(axis=0)
            sd_normalized = normalized_stack.std(axis=0, ddof=1)
            png_name = f"confusion_{dataset.key}_{model.label.lower().replace('-', '_')}.png"
            plot_confusion(
                mean_normalized,
                class_names or [],
                f"{model.label} (mean of 3 seeds)",
                args.out / png_name,
            )
            confusion_result[model.label] = {
                "class_names": class_names,
                "columns": [*(class_names or []), "Miss/FN"],
                "counts_per_seed": matrix_stack.tolist(),
                "normalized_mean": mean_normalized.tolist(),
                "normalized_sample_sd": sd_normalized.tolist(),
                "false_positive_counts_per_seed": fp_stack.tolist(),
                "false_positive_mean": fp_stack.mean(axis=0).tolist(),
                "false_positive_sample_sd": fp_stack.std(axis=0, ddof=1).tolist(),
                "figure": str((args.out / png_name).resolve()),
            }
            for row_index, true_name in enumerate(class_names or []):
                for column_index, predicted_name in enumerate([*(class_names or []), "Miss/FN"]):
                    confusion_csv.append(
                        {
                            "dataset": dataset.label,
                            "model": model.label,
                            "true_class": true_name,
                            "predicted_class": predicted_name,
                            "normalized_mean_percent": mean_normalized[row_index, column_index] * 100.0,
                            "normalized_sample_sd_percent": sd_normalized[row_index, column_index] * 100.0,
                            "seed0_count": matrix_stack[0, row_index, column_index],
                            "seed1_count": matrix_stack[1, row_index, column_index],
                            "seed2_count": matrix_stack[2, row_index, column_index],
                        }
                    )
        dataset_result["confusion"] = confusion_result
        results["datasets"][dataset.key] = dataset_result

    np.savez_compressed(args.out / "bootstrap_distributions.npz", **distributions)
    write_csv(args.out / "bootstrap_ci.csv", bootstrap_csv)
    write_csv(args.out / "dataset_a_area_ap.csv", area_a_csv)
    write_csv(args.out / "dataset_b_area_ap.csv", area_b_csv)
    write_csv(args.out / "dataset_b_class_area_ap.csv", area_b_class_csv)
    write_csv(args.out / "confusion_matrices.csv", confusion_csv)
    (args.out / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote results to {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
