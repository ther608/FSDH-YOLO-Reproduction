"""Create Dataset B v4 train-only augmentation aligned with Dataset A."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


SPLITS = ("train", "val", "test")
CLASS_NAMES = ("green", "half", "red", "young")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_SOURCE = Path(r"<data-root>/data\lichi-maturity\lichi-maturity\lichi_701515_groupaware_clean_v4_raw")
DEFAULT_OUTPUT = Path(r"<data-root>/data\lichi-maturity\lichi-maturity\lichi_701515_groupaware_clean_v4_aug")
SEED = 20260731


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, data = cv2.imencode(path.suffix.lower(), image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    data.tofile(str(path))


def read_labels(path: Path) -> list[list[float]]:
    labels: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields")
        class_id = int(fields[0])
        if class_id not in range(len(CLASS_NAMES)):
            raise ValueError(f"{path}:{line_number}: invalid class {class_id}")
        labels.append([class_id, *map(float, fields[1:])])
    return labels


def write_labels(path: Path, labels: list[list[float]]) -> None:
    lines: list[str] = []
    for class_id, x, y, width, height in labels:
        x, y = np.clip([x, y], 0.0, 1.0)
        width, height = np.clip([width, height], 0.0, 1.0)
        if width > 0 and height > 0:
            lines.append(f"{int(class_id)} {x:.6f} {y:.6f} {width:.6f} {height:.6f}")
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def labels_to_boxes(labels: list[list[float]], width: int, height: int) -> np.ndarray:
    boxes = [
        [class_id, (x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height]
        for class_id, x, y, w, h in labels
    ]
    return np.asarray(boxes, dtype=np.float32)


def boxes_to_labels(boxes: np.ndarray, width: int, height: int) -> list[list[float]]:
    labels: list[list[float]] = []
    for class_id, x1, y1, x2, y2 in boxes:
        x1, y1 = float(np.clip(x1, 0, width - 1)), float(np.clip(y1, 0, height - 1))
        x2, y2 = float(np.clip(x2, 0, width - 1)), float(np.clip(y2, 0, height - 1))
        box_width, box_height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if box_width < 2.0 or box_height < 2.0:
            continue
        labels.append(
            [
                int(class_id),
                (x1 + x2) / (2 * width),
                (y1 + y2) / (2 * height),
                box_width / width,
                box_height / height,
            ]
        )
    return labels


def horizontal_flip(image: np.ndarray, labels: list[list[float]]) -> tuple[np.ndarray, list[list[float]]]:
    return cv2.flip(image, 1), [[class_id, 1.0 - x, y, w, h] for class_id, x, y, w, h in labels]


def affine_transform(
    image: np.ndarray,
    labels: list[list[float]],
    *,
    angle: float = 0.0,
    scale: float = 1.0,
    tx_fraction: float = 0.0,
    ty_fraction: float = 0.0,
) -> tuple[np.ndarray, list[list[float]]]:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[0, 2] += tx_fraction * width
    matrix[1, 2] += ty_fraction * height
    transformed = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    boxes = labels_to_boxes(labels, width, height)
    if boxes.size == 0:
        return transformed, labels
    new_boxes = []
    for class_id, x1, y1, x2, y2 in boxes:
        corners = np.asarray([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float32)
        transformed_corners = corners @ matrix.T
        nx1, ny1 = transformed_corners.min(axis=0)
        nx2, ny2 = transformed_corners.max(axis=0)
        original_area = max(1.0, (x2 - x1) * (y2 - y1))
        clipped_width = max(0.0, min(nx2, width - 1) - max(nx1, 0))
        clipped_height = max(0.0, min(ny2, height - 1) - max(ny1, 0))
        if clipped_width * clipped_height / original_area >= 0.25:
            new_boxes.append([class_id, nx1, ny1, nx2, ny2])
    return transformed, boxes_to_labels(np.asarray(new_boxes, dtype=np.float32), width, height)


def illumination_colour(image: np.ndarray, rng: random.Random) -> tuple[np.ndarray, str, dict[str, float]]:
    if rng.random() < 0.5:
        alpha = rng.uniform(0.88, 1.12)
        beta = rng.uniform(-18.0, 18.0)
        output = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return output, "brightness_contrast", {"contrast_alpha": alpha, "brightness_beta": beta}
    saturation = rng.uniform(0.86, 1.14)
    hue_shift = rng.uniform(-4.0, 4.0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= saturation
    hsv[..., 0] = np.mod(hsv[..., 0] + hue_shift, 180)
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    return (
        cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR),
        "hsv_colour",
        {"saturation_scale": saturation, "hue_shift_opencv_h_units": hue_shift},
    )


def image_quality(image: np.ndarray, rng: random.Random) -> tuple[np.ndarray, str, dict[str, float]]:
    if rng.random() < 0.5:
        sigma = rng.uniform(0.35, 0.85)
        return cv2.GaussianBlur(image, (3, 3), sigma), "gaussian_blur", {"sigma": sigma, "kernel": 3}
    stddev = rng.uniform(3.0, 7.0)
    generator = np.random.default_rng(rng.randrange(1 << 30))
    noise = generator.normal(0.0, stddev, image.shape)
    output = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return output, "sensor_noise", {"gaussian_noise_stddev": stddev}


def mild_geometry(
    image: np.ndarray, labels: list[list[float]], rng: random.Random
) -> tuple[np.ndarray, list[list[float]], str, dict[str, object]]:
    choice = rng.choice(("horizontal_flip", "rotation", "translation", "scale"))
    if choice == "horizontal_flip":
        output, output_labels = horizontal_flip(image, labels)
        return output, output_labels, choice, {"axis": "horizontal"}
    if choice == "rotation":
        angle = rng.uniform(-5.0, 5.0)
        output, output_labels = affine_transform(image, labels, angle=angle)
        return output, output_labels, choice, {"angle_degrees": angle}
    if choice == "translation":
        tx, ty = rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02)
        output, output_labels = affine_transform(image, labels, tx_fraction=tx, ty_fraction=ty)
        return output, output_labels, choice, {"translate_x_fraction": tx, "translate_y_fraction": ty}
    scale = rng.uniform(1.04, 1.10)
    output, output_labels = affine_transform(image, labels, scale=scale)
    return output, output_labels, choice, {"scale": scale}


def safe_remove_building(path: Path, output: Path) -> None:
    resolved, target = path.resolve(), output.resolve()
    if resolved.parent != target.parent or not resolved.name.startswith(f"{target.name}.building_"):
        raise RuntimeError(f"Refusing to remove unexpected temporary path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def count_dataset(root: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        images = [path for path in (root / split / "images").iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
        counts = Counter()
        empty = 0
        for label in (root / split / "labels").glob("*.txt"):
            labels = read_labels(label)
            if not labels:
                empty += 1
            counts.update(int(row[0]) for row in labels)
        result[split] = {
            "images": len(images),
            "objects": sum(counts.values()),
            "empty_labels": empty,
            "class_instances": {CLASS_NAMES[index]: counts[index] for index in range(len(CLASS_NAMES))},
        }
    return result


def main() -> None:
    args = parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Raw v4 dataset not found: {source}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    raw_summary_path = source / "build_summary.json"
    raw_summary = json.loads(raw_summary_path.read_text(encoding="utf-8"))
    if raw_summary.get("schema") != "dataset_b_groupaware_clean_v4_raw":
        raise ValueError("Source is not a verified Dataset B v4 raw build")

    building = output.with_name(f"{output.name}.building_{os.getpid()}")
    if building.exists():
        raise FileExistsError(f"Temporary path exists: {building}")
    rng = random.Random(SEED)
    log_rows: list[dict[str, object]] = []
    materialization = Counter()
    categories = Counter()
    operations = Counter()
    try:
        for split in SPLITS:
            (building / split / "images").mkdir(parents=True)
            (building / split / "labels").mkdir(parents=True)
            source_images = source / split / "images"
            source_labels = source / split / "labels"
            for image_path in sorted(path for path in source_images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
                label_path = source_labels / f"{image_path.stem}.txt"
                if not label_path.is_file():
                    raise FileNotFoundError(f"Missing source label: {label_path}")
                for origin, target in (
                    (image_path, building / split / "images" / image_path.name),
                    (label_path, building / split / "labels" / label_path.name),
                ):
                    try:
                        os.link(origin, target)
                        materialization["hardlinked"] += 1
                    except OSError:
                        shutil.copy2(origin, target)
                        materialization["copied"] += 1
                if split != "train":
                    continue
                image = read_image(image_path)
                labels = read_labels(label_path)
                source_objects = len(labels)
                illum_image, illum_operation, illum_params = illumination_colour(image, rng)
                quality_image, quality_operation, quality_params = image_quality(image, rng)
                geometry_image, geometry_labels, geometry_operation, geometry_params = mild_geometry(image, labels, rng)
                variants = (
                    (1, "illum_color", illum_operation, illum_params, illum_image, labels),
                    (2, "image_quality", quality_operation, quality_params, quality_image, labels),
                    (3, "mild_geometry", geometry_operation, geometry_params, geometry_image, geometry_labels),
                )
                for variant, category, operation, parameters, out_image_data, out_labels in variants:
                    stem = f"{image_path.stem}_aug{variant}_{category}_{operation}"
                    out_image = building / "train" / "images" / f"{stem}{image_path.suffix.lower()}"
                    out_label = building / "train" / "labels" / f"{stem}.txt"
                    write_image(out_image, out_image_data)
                    write_labels(out_label, out_labels)
                    categories[category] += 1
                    operations[operation] += 1
                    log_rows.append(
                        {
                            "source_image": image_path.name,
                            "output_image": out_image.name,
                            "output_label": out_label.name,
                            "variant_index": variant,
                            "category": category,
                            "operation": operation,
                            "parameters": json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                            "source_objects": source_objects,
                            "output_objects": len(out_labels),
                        }
                    )
        root_text = str(output).replace("\\", "/")
        yaml_path = building / "lichi_maturity_groupaware_clean_v4_aug.yaml"
        yaml_path.write_text(
            "# Dataset B v4: Dataset-A-aligned raw split and train-only three-category augmentation.\n"
            f'path: "{root_text}"\n'
            "train: train/images\nval: val/images\ntest: test/images\n\n"
            "nc: 4\nnames:\n  0: green\n  1: half\n  2: red\n  3: young\n",
            encoding="utf-8",
        )
        write_fields = (
            "source_image", "output_image", "output_label", "variant_index", "category", "operation",
            "parameters", "source_objects", "output_objects",
        )
        with (building / "augmentation_log.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=write_fields)
            writer.writeheader()
            writer.writerows(log_rows)
        shutil.copy2(raw_summary_path, building / "raw_build_summary.json")
        raw_stats = count_dataset(source)
        aug_stats = count_dataset(building)
        for split in ("val", "test"):
            for subdir in ("images", "labels"):
                raw_hashes = {
                    path.name: digest(path) for path in (source / split / subdir).iterdir() if path.is_file()
                }
                aug_hashes = {
                    path.name: digest(path) for path in (building / split / subdir).iterdir() if path.is_file()
                }
                if raw_hashes != aug_hashes:
                    raise AssertionError(f"{split}/{subdir} changed during augmentation")
        expected = raw_stats["train"]["images"]
        if categories != Counter({"illum_color": expected, "image_quality": expected, "mild_geometry": expected}):
            raise AssertionError(f"Unexpected category counts: {categories}")
        summary = {
            "schema": "dataset_b_groupaware_clean_v4_aug",
            "source_raw": str(source),
            "output": str(output),
            "seed": SEED,
            "augmentation_scope": "train only",
            "generation_rule": "each training original plus one independent variant from each of three categories",
            "chained_augmentation": False,
            "vertical_flip": False,
            "raw_build_summary_sha256": digest(raw_summary_path),
            "augmentation_log_rows": len(log_rows),
            "category_counts": dict(categories),
            "operation_counts": dict(operations),
            "raw_statistics": raw_stats,
            "augmented_statistics": aug_stats,
            "validation_test_byte_identical_to_raw": True,
            "image_materialization": dict(materialization),
        }
        (building / "augmentation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        building.replace(output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"[CREATED] {output}")
        print(f"[YAML] {output / yaml_path.name}")
    except Exception:
        if building.exists():
            safe_remove_building(building, output)
        raise


if __name__ == "__main__":
    main()
