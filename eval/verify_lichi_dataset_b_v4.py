"""Independently verify Dataset B v4 raw and augmented builds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from audit_lichi_dataset_b_v4 import (
    CLASS_NAMES,
    IMAGE_SUFFIXES,
    NEAR_GRAY32_MAE_MAX,
    NEAR_PHASH_MAX,
    SEQUENTIAL_GAP_MAX,
    SEQUENTIAL_GRAY32_MAE_MAX,
    candidate_groups,
    image_signature,
    is_offline_augmentation,
    parse_label,
    sequence_identity,
)


SPLITS = ("train", "val", "test")
DEFAULT_RAW = Path(r"<data-root>/data\lichi-maturity\lichi-maturity\lichi_701515_groupaware_clean_v4_raw")
DEFAULT_AUG = Path(r"<data-root>/data\lichi-maturity\lichi-maturity\lichi_701515_groupaware_clean_v4_aug")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "analysis" / "dataset_b_v4_verification.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--aug", type=Path, default=DEFAULT_AUG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def files_by_name(directory: Path) -> dict[str, Path]:
    return {path.name: path for path in directory.iterdir() if path.is_file()}


def verify_dataset_pairs(root: Path) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    report: dict[str, dict[str, object]] = {}
    records: list[dict[str, object]] = []
    for split in SPLITS:
        images = sorted(
            path for path in (root / split / "images").iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        labels = {path.stem: path for path in (root / split / "labels").glob("*.txt")}
        if {path.stem for path in images} != set(labels):
            raise AssertionError(f"Image/label mismatch in {root}, split={split}")
        counts = Counter()
        empty = 0
        duplicate_rows = 0
        high_iou_groups = 0
        for image in images:
            label = labels[image.stem]
            raw_lines = [line.strip() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            duplicate_rows += len(raw_lines) - len(dict.fromkeys(raw_lines))
            annotations = parse_label(label)
            high_iou_groups += len(candidate_groups(annotations))
            if not annotations:
                empty += 1
            counts.update(annotation.class_id for annotation in annotations)
            phash, gray32 = image_signature(image)
            family, value = sequence_identity(image.name)
            records.append(
                {
                    "split": split,
                    "image": image,
                    "sha256": digest(image),
                    "phash": phash,
                    "gray32": gray32,
                    "family": family,
                    "value": value,
                }
            )
        report[split] = {
            "images": len(images),
            "objects": sum(counts.values()),
            "empty_labels": empty,
            "duplicate_label_rows": duplicate_rows,
            "residual_iou_gte_090_groups": high_iou_groups,
            "class_instances": {CLASS_NAMES[index]: counts[index] for index in range(len(CLASS_NAMES))},
        }
    return report, records


def verify_raw_leakage(raw: Path, records: list[dict[str, object]]) -> dict[str, object]:
    hashes = defaultdict(set)
    for record in records:
        hashes[record["sha256"]].add(record["split"])
    cross_exact = sum(len(splits) > 1 for splits in hashes.values())
    near_edges = 0
    sequential_edges = 0
    cross_near = 0
    cross_sequential = 0
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            hamming = (int(left["phash"]) ^ int(right["phash"])).bit_count()
            same_sequence = (
                left["family"] != "other"
                and left["family"] == right["family"]
                and left["value"] is not None
                and right["value"] is not None
            )
            gap = abs(int(left["value"]) - int(right["value"])) if same_sequence else None
            near_possible = hamming <= NEAR_PHASH_MAX
            sequence_possible = same_sequence and gap is not None and gap <= SEQUENTIAL_GAP_MAX
            if not near_possible and not sequence_possible:
                continue
            mae = float(np.abs(left["gray32"] - right["gray32"]).mean())
            if near_possible and mae <= NEAR_GRAY32_MAE_MAX:
                near_edges += 1
                cross_near += left["split"] != right["split"]
            if sequence_possible and mae <= SEQUENTIAL_GRAY32_MAE_MAX:
                sequential_edges += 1
                cross_sequential += left["split"] != right["split"]
    with (raw / "split_manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[row["group_id"]].add(row["new_split"])
    cross_groups = sum(len(values) > 1 for values in groups.values())
    offline_remaining = sum(is_offline_augmentation(Path(record["image"]).name) for record in records)
    if cross_exact or cross_near or cross_sequential or cross_groups or offline_remaining:
        raise AssertionError(
            "Raw leakage verification failed: "
            f"exact={cross_exact}, near={cross_near}, sequential={cross_sequential}, "
            f"groups={cross_groups}, offline={offline_remaining}"
        )
    return {
        "cross_split_exact_sha256_groups": cross_exact,
        "recomputed_near_edges": near_edges,
        "recomputed_cross_split_near_edges": cross_near,
        "recomputed_sequential_edges": sequential_edges,
        "recomputed_cross_split_sequential_edges": cross_sequential,
        "cross_split_manifest_groups": cross_groups,
        "offline_augmentations_remaining": offline_remaining,
    }


def validate_parameters(row: dict[str, str]) -> None:
    parameters = json.loads(row["parameters"])
    operation = row["operation"]
    if operation == "brightness_contrast":
        assert 0.88 <= parameters["contrast_alpha"] <= 1.12
        assert -18.0 <= parameters["brightness_beta"] <= 18.0
    elif operation == "hsv_colour":
        assert 0.86 <= parameters["saturation_scale"] <= 1.14
        assert -4.0 <= parameters["hue_shift_opencv_h_units"] <= 4.0
    elif operation == "gaussian_blur":
        assert parameters["kernel"] == 3 and 0.35 <= parameters["sigma"] <= 0.85
    elif operation == "sensor_noise":
        assert 3.0 <= parameters["gaussian_noise_stddev"] <= 7.0
    elif operation == "rotation":
        assert -5.0 <= parameters["angle_degrees"] <= 5.0
    elif operation == "translation":
        assert -0.02 <= parameters["translate_x_fraction"] <= 0.02
        assert -0.02 <= parameters["translate_y_fraction"] <= 0.02
    elif operation == "scale":
        assert 1.04 <= parameters["scale"] <= 1.10
    elif operation == "horizontal_flip":
        assert parameters["axis"] == "horizontal"
    else:
        raise AssertionError(f"Unknown augmentation operation: {operation}")


def verify_augmentation(raw: Path, aug: Path, raw_stats: dict[str, dict[str, object]]) -> dict[str, object]:
    for split in SPLITS:
        for subdir in ("images", "labels"):
            raw_files = files_by_name(raw / split / subdir)
            aug_files = files_by_name(aug / split / subdir)
            for name, source_path in raw_files.items():
                if name not in aug_files or digest(source_path) != digest(aug_files[name]):
                    raise AssertionError(f"Original changed or missing in augmented dataset: {split}/{subdir}/{name}")
            if split in {"val", "test"} and set(raw_files) != set(aug_files):
                raise AssertionError(f"Unexpected augmented files in {split}/{subdir}")
    with (aug / "augmentation_log.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected_train = int(raw_stats["train"]["images"])
    if len(rows) != expected_train * 3:
        raise AssertionError(f"Unexpected augmentation log rows: {len(rows)}")
    per_source: dict[str, set[str]] = defaultdict(set)
    categories = Counter()
    operations = Counter()
    geometry_losses = 0
    for row in rows:
        validate_parameters(row)
        per_source[row["source_image"]].add(row["category"])
        categories[row["category"]] += 1
        operations[row["operation"]] += 1
        geometry_losses += int(row["source_objects"]) - int(row["output_objects"])
        image = aug / "train" / "images" / row["output_image"]
        label = aug / "train" / "labels" / row["output_label"]
        if not image.is_file() or not label.is_file():
            raise AssertionError(f"Logged augmentation missing: {row}")
    required = {"illum_color", "image_quality", "mild_geometry"}
    if len(per_source) != expected_train or any(values != required for values in per_source.values()):
        raise AssertionError("Each training original must have exactly the three required categories")
    return {
        "augmentation_log_rows": len(rows),
        "sources_with_three_categories": len(per_source),
        "category_counts": dict(categories),
        "operation_counts": dict(operations),
        "geometry_boxes_removed_by_visibility_or_min_size_rules": geometry_losses,
        "raw_originals_byte_identical_in_augmented_dataset": True,
        "validation_test_have_no_generated_images": True,
    }


def main() -> None:
    args = parse_args()
    raw, aug, output = args.raw.resolve(), args.aug.resolve(), args.output.resolve()
    raw_stats, records = verify_dataset_pairs(raw)
    if any(values["duplicate_label_rows"] or values["residual_iou_gte_090_groups"] for values in raw_stats.values()):
        raise AssertionError(f"Residual duplicate boxes remain: {raw_stats}")
    leakage = verify_raw_leakage(raw, records)
    aug_stats, _ = verify_dataset_pairs(aug)
    augmentation = verify_augmentation(raw, aug, raw_stats)
    report = {
        "status": "PASS",
        "schema": "dataset_b_v4_independent_verification_v1",
        "raw": str(raw),
        "augmented": str(aug),
        "raw_statistics": raw_stats,
        "augmented_statistics": aug_stats,
        "leakage_verification": leakage,
        "augmentation_verification": augmentation,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
