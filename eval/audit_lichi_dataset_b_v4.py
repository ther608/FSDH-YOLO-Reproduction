"""Audit Dataset B under the protocol frozen from Dataset A.

This script is intentionally model-blind. It reads only source images, YOLO
labels, and filename-derived acquisition metadata. It does not read model
predictions, validation metrics, test metrics, or prior split scores.

The output is an audit/review package, not a dataset. A final v4 builder must
consume completed human decisions for every high-IoU box group and every empty
label image before materializing a split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


CLASS_NAMES = ("green", "half", "red", "young")
CLASS_COLORS = ((50, 210, 70), (255, 205, 30), (245, 60, 60), (70, 170, 255))
SOURCE_SPLITS = ("train", "val")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SOURCE = Path(r"<data-root>/data\lichi-maturity\lichi-maturity\lichi")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "analysis" / "dataset_b_v4_protocol_audit"

ROBOFLOW_SUFFIX = re.compile(r"_(?:jpg|png)\.rf\.[0-9a-f]+\.(?:jpg|jpeg|png)$", re.IGNORECASE)
OFFLINE_AUGMENTATION = re.compile(
    r"^(?:flipped_horizontal_|flipped_vertical_|rotated_-15_|rotated_15_)", re.IGNORECASE
)

# Frozen from Dataset A. These defaults must not be tuned using model results.
BOX_IOU_THRESHOLD = 0.90
NEAR_PHASH_MAX = 10
NEAR_GRAY32_MAE_MAX = 0.06
SEQUENTIAL_GAP_MAX = 3
SEQUENTIAL_GRAY32_MAE_MAX = 0.10
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 20260731
SPLIT_TRIALS = 1000


@dataclass(frozen=True)
class Annotation:
    index: int
    line_no: int
    class_id: int
    x: float
    y: float
    w: float
    h: float
    raw: str

    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x - self.w / 2, self.y - self.h / 2, self.x + self.w / 2, self.y + self.h / 2


@dataclass(frozen=True)
class ImageRecord:
    image: Path
    label: Path
    legacy_split: str
    normalized_origin: str
    offline_augmentation: bool
    sha256: str
    phash: int
    gray32: np.ndarray
    sequence_family: str
    sequence_value: int | None


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        if self.parent[index] != index:
            self.parent[index] = self.find(self.parent[index])
        return self.parent[index]

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def roboflow_base(filename: str) -> str:
    return ROBOFLOW_SUFFIX.sub("", filename)


def normalized_origin(filename: str) -> str:
    return OFFLINE_AUGMENTATION.sub("", roboflow_base(filename))


def is_offline_augmentation(filename: str) -> bool:
    return OFFLINE_AUGMENTATION.match(roboflow_base(filename)) is not None


def sequence_identity(filename: str) -> tuple[str, int | None]:
    base = normalized_origin(filename)
    match = re.fullmatch(r"-(\d+)-", base)
    if match:
        return "dash", int(match.group(1))
    match = re.fullmatch(r"Lychee_(\d+)", base, flags=re.IGNORECASE)
    if match:
        return "lychee", int(match.group(1))
    match = re.fullmatch(r"IMG_(\d{8})_(\d{6})(?:_\d+)?", base, flags=re.IGNORECASE)
    if match:
        hhmmss = match.group(2)
        seconds = int(hhmmss[:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:])
        return f"img_{match.group(1)}", seconds
    return "other", None


def image_signature(path: Path) -> tuple[int, np.ndarray]:
    with Image.open(path) as opened:
        gray32 = np.asarray(
            ImageOps.exif_transpose(opened).convert("L").resize((32, 32), Image.Resampling.LANCZOS),
            dtype=np.float32,
        ) / 255.0
    n = 32
    k = np.arange(n, dtype=np.float32)[:, None]
    x = np.arange(n, dtype=np.float32)[None, :]
    basis = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    dct = basis @ gray32 @ basis.T
    low = dct[:8, :8].reshape(-1)[1:]
    median = float(np.median(low))
    phash = 0
    for bit in low > median:
        phash = (phash << 1) | int(bit)
    return phash, gray32


def parse_label(path: Path) -> list[Annotation]:
    annotations: list[Annotation] = []
    for index, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
        stripped = raw.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{index + 1}: expected 5 YOLO fields, found {len(fields)}")
        class_value = float(fields[0])
        class_id = int(class_value)
        if class_value != class_id or class_id not in range(len(CLASS_NAMES)):
            raise ValueError(f"{path}:{index + 1}: invalid class ID {fields[0]}")
        x, y, width, height = map(float, fields[1:])
        values = (x, y, width, height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{index + 1}: non-finite box")
        if not all(0.0 <= value <= 1.0 for value in values) or width <= 0 or height <= 0:
            raise ValueError(f"{path}:{index + 1}: invalid normalized box")
        annotations.append(Annotation(len(annotations), index + 1, class_id, x, y, width, height, stripped))
    return annotations


def box_iou(left: Annotation, right: Annotation) -> float:
    lx1, ly1, lx2, ly2 = left.xyxy()
    rx1, ry1, rx2, ry2 = right.xyxy()
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(0.0, min(ly2, ry2) - max(ly1, ry1))
    union = left.w * left.h + right.w * right.h - intersection
    return intersection / union if union > 0 else 0.0


def candidate_groups(annotations: list[Annotation]) -> list[dict[str, object]]:
    dsu = DisjointSet(len(annotations))
    pairs: list[tuple[int, int, float]] = []
    for left in range(len(annotations)):
        for right in range(left + 1, len(annotations)):
            iou = box_iou(annotations[left], annotations[right])
            if iou >= BOX_IOU_THRESHOLD:
                dsu.union(left, right)
                pairs.append((left, right, iou))
    members: dict[int, set[int]] = defaultdict(set)
    for left, right, _ in pairs:
        root = dsu.find(left)
        members[root].update((left, right))
    groups: list[dict[str, object]] = []
    for indices in sorted(members.values(), key=lambda values: min(values)):
        group_pairs = [pair for pair in pairs if pair[0] in indices and pair[1] in indices]
        groups.append(
            {
                "annotation_indices": sorted(indices),
                "max_iou": max(pair[2] for pair in group_pairs),
                "min_iou": min(pair[2] for pair in group_pairs),
                "class_relation": (
                    "same-class"
                    if len({annotations[index].class_id for index in indices}) == 1
                    else "cross-class"
                ),
            }
        )
    return groups


def annotation_pixels(annotation: Annotation, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = annotation.xyxy()
    return (
        round(max(0.0, min(1.0, x1)) * width),
        round(max(0.0, min(1.0, y1)) * height),
        round(max(0.0, min(1.0, x2)) * width),
        round(max(0.0, min(1.0, y2)) * height),
    )


def render_group(
    image_path: Path,
    annotations: list[Annotation],
    indices: list[int],
    output_path: Path,
) -> None:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    width, height = image.size
    boxes = [annotation_pixels(annotations[index], width, height) for index in indices]
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    padding = max(40, round(max(x2 - x1, y2 - y1) * 1.4))
    bounds = (max(0, x1 - padding), max(0, y1 - padding), min(width, x2 + padding), min(height, y2 + padding))
    crop = image.crop(bounds)
    draw = ImageDraw.Draw(crop)
    font = ImageFont.load_default()
    for rank, index in enumerate(indices):
        annotation = annotations[index]
        bx1, by1, bx2, by2 = boxes[rank]
        bx1, by1, bx2, by2 = bx1 - bounds[0], by1 - bounds[1], bx2 - bounds[0], by2 - bounds[1]
        color = CLASS_COLORS[annotation.class_id]
        draw.rectangle((bx1, by1, bx2, by2), outline=color, width=4 + 3 * (len(indices) - rank - 1))
        label = f"L{annotation.line_no} {annotation.class_id}:{CLASS_NAMES[annotation.class_id]}"
        text_box = draw.textbbox((bx1, by1), label, font=font, stroke_width=1)
        text_width = text_box[2] - text_box[0] + 6
        text_height = text_box[3] - text_box[1] + 6
        text_y = max(0, by1 - text_height * (rank + 1))
        draw.rectangle((bx1, text_y, bx1 + text_width, text_y + text_height), fill=color)
        draw.text((bx1 + 3, text_y + 2), label, fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)


def render_empty_label(image_path: Path, output_path: Path) -> None:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    image.thumbnail((1200, 900), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 36), fill=(255, 230, 80))
    draw.text((8, 10), "EMPTY LABEL: keep only if no target fruit is present", fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def protocol_lock(source: Path) -> dict[str, object]:
    return {
        "schema": "dataset_b_v4_protocol_lock_v1",
        "source": str(source.resolve()),
        "protocol_origin": "Dataset A frozen data-cleaning and group-aware split rules",
        "allowed_inputs": ["source images", "YOLO labels", "filename-derived acquisition metadata"],
        "prohibited_inputs": [
            "model predictions",
            "validation metrics",
            "test metrics",
            "prior experiment rankings",
        ],
        "offline_augmentation_policy": (
            "exclude filename-identifiable source-provided flips/rotations before grouping; "
            "any v4 augmentation must be generated after the split and only from training originals"
        ),
        "box_review": {
            "scope": "all classes within the same image",
            "iou_threshold_gte": BOX_IOU_THRESHOLD,
            "automatic_deletion": False,
            "required_human_actions": ["keep_one with source line number", "keep_all"],
        },
        "image_grouping": {
            "exact_sha256": "group, do not auto-delete images",
            "near_duplicate": {
                "phash_hamming_lte": NEAR_PHASH_MAX,
                "gray32_normalized_mae_lte": NEAR_GRAY32_MAE_MAX,
            },
            "same_sequence": {
                "filename_derived_gap_lte": SEQUENTIAL_GAP_MAX,
                "gray32_normalized_mae_lte": SEQUENTIAL_GRAY32_MAE_MAX,
                "phash_condition": None,
            },
            "component_policy": "every connected component is an indivisible split unit",
        },
        "split": {
            "ratios": SPLIT_RATIOS,
            "seed": SPLIT_SEED,
            "trials": SPLIT_TRIALS,
            "balanced_features": ["image count", "object count", *CLASS_NAMES],
            "relative_squared_error_weights": {
                "image_count": 5.0,
                "object_count": 1.0,
                "each_class_count": 2.0,
            },
        },
    }


def main() -> None:
    args = parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit package: {output}")
    output.mkdir(parents=True)
    assets = output / "review_assets"

    protocol = protocol_lock(source)
    (output / "protocol_lock.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    records: list[ImageRecord] = []
    excluded_offline: list[dict[str, object]] = []
    box_manifest: list[dict[str, object]] = []
    empty_manifest: list[dict[str, object]] = []
    counts_before = Counter()
    image_names: set[str] = set()

    for legacy_split in SOURCE_SPLITS:
        image_dir = source / legacy_split / "images"
        label_dir = source / legacy_split / "labels"
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        stems = {path.stem for path in images}
        orphan_labels = sorted(path.name for path in label_dir.glob("*.txt") if path.stem not in stems)
        if orphan_labels:
            raise ValueError(f"{legacy_split}: orphan labels found, first={orphan_labels[0]}")
        for image_path in images:
            if image_path.name in image_names:
                raise ValueError(f"Duplicate filename across legacy splits: {image_path.name}")
            image_names.add(image_path.name)
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label: {label_path}")
            annotations = parse_label(label_path)
            counts_before.update(annotation.class_id for annotation in annotations)
            if is_offline_augmentation(image_path.name):
                excluded_offline.append(
                    {
                        "file": image_path.name,
                        "legacy_split": legacy_split,
                        "normalized_origin": normalized_origin(image_path.name),
                        "reason": "filename_identified_source_offline_flip_or_rotation",
                    }
                )
                continue

            groups = candidate_groups(annotations)
            for group_number, group in enumerate(groups, 1):
                indices = list(group["annotation_indices"])
                group_id = f"{image_path.stem}/G{group_number:03d}"
                asset_name = f"box_{len(box_manifest) + 1:03d}.png"
                render_group(image_path, annotations, indices, assets / asset_name)
                box_manifest.append(
                    {
                        "group_id": group_id,
                        "legacy_split": legacy_split,
                        "image": str(image_path),
                        "label": str(label_path),
                        "asset": f"review_assets/{asset_name}",
                        "class_relation": group["class_relation"],
                        "max_iou": round(float(group["max_iou"]), 6),
                        "min_iou": round(float(group["min_iou"]), 6),
                        "annotations": [asdict(annotations[index]) for index in indices],
                    }
                )
            if not annotations:
                empty_id = image_path.stem
                asset_name = f"empty_{len(empty_manifest) + 1:03d}.png"
                render_empty_label(image_path, assets / asset_name)
                empty_manifest.append(
                    {
                        "image_id": empty_id,
                        "legacy_split": legacy_split,
                        "image": str(image_path),
                        "label": str(label_path),
                        "asset": f"review_assets/{asset_name}",
                    }
                )

            phash, gray32 = image_signature(image_path)
            sequence_family, sequence_value = sequence_identity(image_path.name)
            records.append(
                ImageRecord(
                    image=image_path,
                    label=label_path,
                    legacy_split=legacy_split,
                    normalized_origin=normalized_origin(image_path.name),
                    offline_augmentation=False,
                    sha256=file_sha256(image_path),
                    phash=phash,
                    gray32=gray32,
                    sequence_family=sequence_family,
                    sequence_value=sequence_value,
                )
            )

    dsu = DisjointSet(len(records))
    edges: list[dict[str, object]] = []
    exact_by_hash: dict[str, int] = {}
    for index, record in enumerate(records):
        if record.sha256 in exact_by_hash:
            other = exact_by_hash[record.sha256]
            dsu.union(other, index)
            edges.append(
                {
                    "edge_type": "exact_sha256",
                    "image_a": str(records[other].image),
                    "image_b": str(record.image),
                    "phash_hamming": 0,
                    "gray32_mae": 0.0,
                    "sequence_gap": "",
                }
            )
        else:
            exact_by_hash[record.sha256] = index

    for left_index, left in enumerate(records):
        for right_index in range(left_index + 1, len(records)):
            right = records[right_index]
            hamming = (left.phash ^ right.phash).bit_count()
            same_sequence = (
                left.sequence_family != "other"
                and left.sequence_family == right.sequence_family
                and left.sequence_value is not None
                and right.sequence_value is not None
            )
            gap = abs(left.sequence_value - right.sequence_value) if same_sequence else None
            near_possible = hamming <= NEAR_PHASH_MAX
            sequence_possible = same_sequence and gap is not None and gap <= SEQUENTIAL_GAP_MAX
            if not near_possible and not sequence_possible:
                continue
            mae = float(np.abs(left.gray32 - right.gray32).mean())
            if near_possible and mae <= NEAR_GRAY32_MAE_MAX:
                dsu.union(left_index, right_index)
                edges.append(
                    {
                        "edge_type": "near_candidate",
                        "image_a": str(left.image),
                        "image_b": str(right.image),
                        "phash_hamming": hamming,
                        "gray32_mae": round(mae, 6),
                        "sequence_gap": gap if gap is not None else "",
                    }
                )
            if sequence_possible and mae <= SEQUENTIAL_GRAY32_MAE_MAX:
                dsu.union(left_index, right_index)
                edges.append(
                    {
                        "edge_type": "sequential_scene_constraint",
                        "image_a": str(left.image),
                        "image_b": str(right.image),
                        "phash_hamming": hamming,
                        "gray32_mae": round(mae, 6),
                        "sequence_gap": gap,
                    }
                )

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        members[dsu.find(index)].append(index)
    groups = list(members.values())

    decisions = {
        "schema": "dataset_b_v4_human_review_v1",
        "box_decisions": {
            group["group_id"]: {"action": "pending", "keep_line": None, "note": ""}
            for group in box_manifest
        },
        "empty_label_decisions": {
            item["image_id"]: {"action": "pending", "note": ""} for item in empty_manifest
        },
        "allowed_box_actions": ["keep_one", "keep_all"],
        "allowed_empty_label_actions": ["keep", "exclude"],
    }
    (output / "box_review_manifest.json").write_text(
        json.dumps(box_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "empty_label_review_manifest.json").write_text(
        json.dumps(empty_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "review_decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(
        output / "excluded_offline_augmentations.csv",
        ("file", "legacy_split", "normalized_origin", "reason"),
        excluded_offline,
    )
    write_csv(
        output / "image_constraint_edges.csv",
        ("edge_type", "image_a", "image_b", "phash_hamming", "gray32_mae", "sequence_gap"),
        edges,
    )

    summary = {
        "schema": "dataset_b_v4_protocol_audit_v1",
        "source": str(source),
        "source_images": len(records) + len(excluded_offline),
        "offline_augmentations_excluded_by_protocol": len(excluded_offline),
        "retained_original_candidates_before_human_review": len(records),
        "box_candidate_groups_iou_gte_090": len(box_manifest),
        "box_candidate_relation_counts": dict(Counter(group["class_relation"] for group in box_manifest)),
        "empty_label_images_requiring_review": len(empty_manifest),
        "class_instances_before_offline_exclusion": {
            CLASS_NAMES[index]: counts_before[index] for index in range(len(CLASS_NAMES))
        },
        "image_constraint_edges": dict(Counter(edge["edge_type"] for edge in edges)),
        "group_count_before_human_review": len(groups),
        "multi_image_group_count": sum(len(group) > 1 for group in groups),
        "largest_group_sizes": sorted((len(group) for group in groups), reverse=True)[:10],
        "human_review_complete": False,
        "final_v4_materialization_allowed": False,
    }
    (output / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Review package: {output}")


if __name__ == "__main__":
    main()
