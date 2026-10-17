"""Build Dataset B v4 using the protocol frozen from Dataset A.

The builder refuses to run while any required human-review decision is
pending. It uses only source images, cleaned labels, filename-derived sequence
metadata, and the locked review package. No model output or metric is read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from audit_lichi_dataset_b_v4 import (
    BOX_IOU_THRESHOLD,
    CLASS_NAMES,
    DisjointSet,
    IMAGE_SUFFIXES,
    NEAR_GRAY32_MAE_MAX,
    NEAR_PHASH_MAX,
    SEQUENTIAL_GAP_MAX,
    SEQUENTIAL_GRAY32_MAE_MAX,
    SOURCE_SPLITS,
    SPLIT_RATIOS,
    SPLIT_SEED,
    SPLIT_TRIALS,
    candidate_groups,
    file_sha256,
    image_signature,
    is_offline_augmentation,
    normalized_origin,
    parse_label,
    sequence_identity,
)


SPLITS = ("train", "val", "test")
DEFAULT_SOURCE = Path(r"<data-root>/data\lichi-maturity\lichi-maturity\lichi")
DEFAULT_REVIEW = Path(__file__).resolve().parents[1] / "analysis" / "dataset_b_v4_protocol_audit"
DEFAULT_OUTPUT = Path(r"<data-root>/data\lichi-maturity\lichi-maturity\lichi_701515_groupaware_clean_v4_raw")


@dataclass(frozen=True)
class CleanRecord:
    image: Path
    legacy_split: str
    label_name: str
    label_lines: tuple[str, ...]
    class_counts: tuple[int, ...]
    sha256: str
    phash: int
    gray32: object
    sequence_family: str
    sequence_value: int | None


@dataclass
class GroupRecord:
    group_id: str
    members: list[int]
    image_count: int
    object_count: int
    class_counts: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def hash_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_review(review: Path) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    decisions_path = review / "review_decisions.json"
    box_manifest = read_json(review / "box_review_manifest.json")
    empty_manifest = read_json(review / "empty_label_review_manifest.json")
    decisions = read_json(decisions_path)
    if not isinstance(decisions, dict) or not isinstance(box_manifest, list) or not isinstance(empty_manifest, list):
        raise ValueError("Malformed review package")
    box_decisions = decisions["box_decisions"]
    empty_decisions = decisions["empty_label_decisions"]
    expected_boxes = {row["group_id"] for row in box_manifest}
    expected_empty = {row["image_id"] for row in empty_manifest}
    if set(box_decisions) != expected_boxes or set(empty_decisions) != expected_empty:
        raise ValueError("Review decision IDs do not exactly match review manifests")
    pending = [key for key, value in box_decisions.items() if value.get("action") == "pending"]
    pending += [key for key, value in empty_decisions.items() if value.get("action") == "pending"]
    if pending:
        raise RuntimeError(f"Human review is incomplete; pending={pending}")
    for group_id, decision in box_decisions.items():
        if decision.get("action") not in {"keep_one", "keep_all"}:
            raise ValueError(f"Invalid box action for {group_id}: {decision}")
        if decision["action"] == "keep_one" and not isinstance(decision.get("keep_line"), int):
            raise ValueError(f"keep_one requires keep_line for {group_id}")
    for image_id, decision in empty_decisions.items():
        if decision.get("action") not in {"keep", "exclude"}:
            raise ValueError(f"Invalid empty-label action for {image_id}: {decision}")
    return box_decisions, empty_decisions


def collect_clean_records(
    source: Path,
    box_decisions: dict[str, dict[str, object]],
    empty_decisions: dict[str, dict[str, object]],
) -> tuple[list[CleanRecord], list[dict[str, object]], list[dict[str, object]]]:
    records: list[CleanRecord] = []
    exclusions: list[dict[str, object]] = []
    cleaning_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for legacy_split in SOURCE_SPLITS:
        image_dir = source / legacy_split / "images"
        label_dir = source / legacy_split / "labels"
        for image in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            if image.name in seen:
                raise ValueError(f"Duplicate filename across legacy splits: {image.name}")
            seen.add(image.name)
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Missing label: {label}")
            annotations = parse_label(label)
            if is_offline_augmentation(image.name):
                exclusions.append(
                    {
                        "file": image.name,
                        "legacy_split": legacy_split,
                        "reason": "filename_identified_source_offline_flip_or_rotation",
                        "normalized_origin": normalized_origin(image.name),
                    }
                )
                continue
            if not annotations:
                decision = empty_decisions.get(image.stem)
                if decision is None:
                    raise KeyError(f"Missing empty-label decision for {image.stem}")
                if decision["action"] == "exclude":
                    exclusions.append(
                        {
                            "file": image.name,
                            "legacy_split": legacy_split,
                            "reason": "human_review_excluded_empty_label_image",
                            "normalized_origin": normalized_origin(image.name),
                        }
                    )
                    continue

            remove_lines: set[int] = set()
            groups = candidate_groups(annotations)
            for group_number, group in enumerate(groups, 1):
                group_id = f"{image.stem}/G{group_number:03d}"
                decision = box_decisions.get(group_id)
                if decision is None:
                    raise KeyError(f"Missing box decision for {group_id}")
                indices = list(group["annotation_indices"])
                candidate_lines = [annotations[index].line_no for index in indices]
                if decision["action"] == "keep_one":
                    keep_line = int(decision["keep_line"])
                    if keep_line not in candidate_lines:
                        raise ValueError(
                            f"Decision line L{keep_line} is not in {group_id}: candidates={candidate_lines}"
                        )
                    remove_lines.update(line for line in candidate_lines if line != keep_line)
                cleaning_rows.append(
                    {
                        "group_id": group_id,
                        "file": image.name,
                        "legacy_split": legacy_split,
                        "class_relation": group["class_relation"],
                        "max_iou": round(float(group["max_iou"]), 6),
                        "candidate_lines": " ".join(map(str, candidate_lines)),
                        "action": decision["action"],
                        "keep_line": decision.get("keep_line") or "",
                        "removed_lines": " ".join(map(str, sorted(line for line in candidate_lines if line in remove_lines))),
                        "note": decision.get("note", ""),
                    }
                )
            cleaned = tuple(annotation.raw for annotation in annotations if annotation.line_no not in remove_lines)
            counts = [0] * len(CLASS_NAMES)
            for line in cleaned:
                counts[int(line.split()[0])] += 1
            phash, gray32 = image_signature(image)
            sequence_family, sequence_value = sequence_identity(image.name)
            records.append(
                CleanRecord(
                    image=image,
                    legacy_split=legacy_split,
                    label_name=label.name,
                    label_lines=cleaned,
                    class_counts=tuple(counts),
                    sha256=file_sha256(image),
                    phash=phash,
                    gray32=gray32,
                    sequence_family=sequence_family,
                    sequence_value=sequence_value,
                )
            )
    return records, exclusions, cleaning_rows


def build_groups(records: list[CleanRecord]) -> tuple[list[GroupRecord], list[dict[str, object]]]:
    dsu = DisjointSet(len(records))
    edges: list[dict[str, object]] = []
    exact: dict[str, int] = {}
    for index, record in enumerate(records):
        if record.sha256 in exact:
            other = exact[record.sha256]
            dsu.union(other, index)
            edges.append(
                {
                    "edge_type": "exact_sha256",
                    "image_a": records[other].image.name,
                    "image_b": record.image.name,
                    "phash_hamming": 0,
                    "gray32_mae": 0.0,
                    "sequence_gap": "",
                }
            )
        else:
            exact[record.sha256] = index
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
            mae = float(abs(left.gray32 - right.gray32).mean())
            if near_possible and mae <= NEAR_GRAY32_MAE_MAX:
                dsu.union(left_index, right_index)
                edges.append(
                    {
                        "edge_type": "near_candidate",
                        "image_a": left.image.name,
                        "image_b": right.image.name,
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
                        "image_a": left.image.name,
                        "image_b": right.image.name,
                        "phash_hamming": hamming,
                        "gray32_mae": round(mae, 6),
                        "sequence_gap": gap,
                    }
                )
    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        members_by_root[dsu.find(index)].append(index)
    groups: list[GroupRecord] = []
    for group_index, members in enumerate(
        sorted(members_by_root.values(), key=lambda values: min(records[index].image.name for index in values)), 1
    ):
        class_counts = tuple(sum(records[index].class_counts[c] for index in members) for c in range(len(CLASS_NAMES)))
        groups.append(
            GroupRecord(
                group_id=f"G{group_index:04d}",
                members=members,
                image_count=len(members),
                object_count=sum(class_counts),
                class_counts=class_counts,
            )
        )
    return groups, edges


def split_objective(stats: dict[str, list[float]], targets: dict[str, list[float]]) -> float:
    score = 0.0
    weights = (5.0, 1.0, *([2.0] * len(CLASS_NAMES)))
    for split in SPLITS:
        for index, weight in enumerate(weights):
            target = max(targets[split][index], 1.0)
            error = (stats[split][index] - target) / target
            score += weight * error * error
            if stats[split][index] > target * 1.12:
                score += weight * 3.0 * error * error
    return score


def assign_groups(groups: list[GroupRecord]) -> tuple[dict[str, str], dict[str, list[float]], float]:
    total = [
        sum(group.image_count for group in groups),
        sum(group.object_count for group in groups),
        *[sum(group.class_counts[c] for group in groups) for c in range(len(CLASS_NAMES))],
    ]
    targets = {split: [value * SPLIT_RATIOS[split] for value in total] for split in SPLITS}
    best_assignment: dict[str, str] | None = None
    best_stats: dict[str, list[float]] | None = None
    best_score = math.inf
    for trial in range(SPLIT_TRIALS):
        rng = random.Random(SPLIT_SEED + trial * 7919)
        ordered = sorted(groups, key=lambda group: (-group.image_count, -group.object_count, rng.random()))
        stats = {split: [0.0] * (2 + len(CLASS_NAMES)) for split in SPLITS}
        assignment: dict[str, str] = {}
        for group in ordered:
            vector = [group.image_count, group.object_count, *group.class_counts]
            choices: list[tuple[float, float, str]] = []
            for split in SPLITS:
                candidate = {name: values.copy() for name, values in stats.items()}
                candidate[split] = [left + right for left, right in zip(candidate[split], vector)]
                choices.append((split_objective(candidate, targets), rng.random(), split))
            chosen = min(choices)[2]
            stats[chosen] = [left + right for left, right in zip(stats[chosen], vector)]
            assignment[group.group_id] = chosen
        score = split_objective(stats, targets)
        valid = all(stats[split][0] > 0 and all(value > 0 for value in stats[split][2:]) for split in SPLITS)
        if valid and score < best_score:
            best_assignment, best_stats, best_score = assignment, stats, score
    if best_assignment is None or best_stats is None:
        raise RuntimeError("No valid group-aware split found")
    return best_assignment, best_stats, best_score


def verify(
    records: list[CleanRecord],
    groups: list[GroupRecord],
    assignment: dict[str, str],
    edges: list[dict[str, object]],
) -> dict[str, object]:
    split_by_image: dict[str, str] = {}
    hashes: dict[str, set[str]] = {split: set() for split in SPLITS}
    for group in groups:
        split = assignment[group.group_id]
        for index in group.members:
            record = records[index]
            split_by_image[record.image.name] = split
            hashes[split].add(record.sha256)
    edge_violations = [
        edge for edge in edges if split_by_image[str(edge["image_a"])] != split_by_image[str(edge["image_b"])]
    ]
    intersections = {
        f"{left}_{right}": len(hashes[left] & hashes[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    if edge_violations or any(intersections.values()):
        raise AssertionError(f"Leakage verification failed: edges={len(edge_violations)}, hashes={intersections}")
    return {
        "constraint_edges_checked": len(edges),
        "constraint_edge_cross_split_violations": 0,
        "group_cross_split_violations": 0,
        "exact_hash_cross_split_counts": intersections,
    }


def safe_remove_building(path: Path, output: Path) -> None:
    resolved, target = path.resolve(), output.resolve()
    if resolved.parent != target.parent or not resolved.name.startswith(f"{target.name}.building_"):
        raise RuntimeError(f"Refusing to remove unexpected temporary path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def materialize(
    output: Path,
    review: Path,
    records: list[CleanRecord],
    exclusions: list[dict[str, object]],
    cleaning_rows: list[dict[str, object]],
    groups: list[GroupRecord],
    edges: list[dict[str, object]],
    assignment: dict[str, str],
    summary: dict[str, object],
) -> Path:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    building = output.with_name(f"{output.name}.building_{os.getpid()}")
    if building.exists():
        raise FileExistsError(f"Temporary path exists: {building}")
    group_by_member: dict[int, GroupRecord] = {}
    for group in groups:
        for member in group.members:
            group_by_member[member] = group
    split_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    materialization = Counter()
    try:
        for split in SPLITS:
            (building / split / "images").mkdir(parents=True)
            (building / split / "labels").mkdir(parents=True)
        for index, record in enumerate(records):
            group = group_by_member[index]
            split = assignment[group.group_id]
            target_image = building / split / "images" / record.image.name
            target_label = building / split / "labels" / record.label_name
            try:
                os.link(record.image, target_image)
                materialization["hardlinked"] += 1
            except OSError:
                shutil.copy2(record.image, target_image)
                materialization["copied"] += 1
            target_label.write_text(
                ("\n".join(record.label_lines) + "\n") if record.label_lines else "", encoding="utf-8"
            )
            split_rows.append(
                {
                    "group_id": group.group_id,
                    "new_split": split,
                    "legacy_split": record.legacy_split,
                    "filename": record.image.name,
                    "objects": sum(record.class_counts),
                    **{f"class_{c}": record.class_counts[c] for c in range(len(CLASS_NAMES))},
                    "image_sha256": record.sha256,
                    "sequence_family": record.sequence_family,
                    "sequence_value": record.sequence_value if record.sequence_value is not None else "",
                }
            )
        for group in groups:
            group_rows.append(
                {
                    "group_id": group.group_id,
                    "split": assignment[group.group_id],
                    "image_count": group.image_count,
                    "object_count": group.object_count,
                    **{f"class_{c}": group.class_counts[c] for c in range(len(CLASS_NAMES))},
                }
            )
        root_text = str(output.resolve()).replace("\\", "/")
        yaml_path = building / "lichi_maturity_groupaware_clean_v4_raw.yaml"
        yaml_path.write_text(
            "# Dataset B v4 raw: Dataset-A-aligned cleaning and group-aware split.\n"
            "# Physiological order: young -> green -> half -> red.\n"
            f'path: "{root_text}"\n'
            "train: train/images\nval: val/images\ntest: test/images\n\n"
            "nc: 4\nnames:\n  0: green\n  1: half\n  2: red\n  3: young\n",
            encoding="utf-8",
        )
        write_csv(
            building / "split_manifest.csv",
            (
                "group_id", "new_split", "legacy_split", "filename", "objects",
                "class_0", "class_1", "class_2", "class_3", "image_sha256",
                "sequence_family", "sequence_value",
            ),
            split_rows,
        )
        write_csv(
            building / "group_manifest.csv",
            ("group_id", "split", "image_count", "object_count", "class_0", "class_1", "class_2", "class_3"),
            group_rows,
        )
        write_csv(
            building / "image_constraint_edges.csv",
            ("edge_type", "image_a", "image_b", "phash_hamming", "gray32_mae", "sequence_gap"),
            edges,
        )
        write_csv(
            building / "label_cleaning_manifest.csv",
            (
                "group_id", "file", "legacy_split", "class_relation", "max_iou", "candidate_lines",
                "action", "keep_line", "removed_lines", "note",
            ),
            cleaning_rows,
        )
        write_csv(
            building / "excluded_images.csv",
            ("file", "legacy_split", "reason", "normalized_origin"),
            exclusions,
        )
        audit_dir = building / "audit"
        audit_dir.mkdir()
        for name in (
            "protocol_lock.json",
            "review_decisions.json",
            "box_review_manifest.json",
            "empty_label_review_manifest.json",
            "audit_summary.json",
        ):
            shutil.copy2(review / name, audit_dir / name)
        summary["image_materialization"] = dict(materialization)
        (building / "build_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        building.replace(output)
        return output / yaml_path.name
    except Exception:
        if building.exists():
            safe_remove_building(building, output)
        raise


def main() -> None:
    args = parse_args()
    source, review, output = args.source.resolve(), args.review.resolve(), args.output.resolve()
    protocol_path = review / "protocol_lock.json"
    decisions_path = review / "review_decisions.json"
    if not protocol_path.is_file() or not decisions_path.is_file():
        raise FileNotFoundError("Review package is incomplete")
    box_decisions, empty_decisions = validate_review(review)
    records, exclusions, cleaning_rows = collect_clean_records(source, box_decisions, empty_decisions)
    groups, edges = build_groups(records)
    assignment, split_stats, objective = assign_groups(groups)
    verification = verify(records, groups, assignment, edges)
    summary: dict[str, object] = {
        "schema": "dataset_b_groupaware_clean_v4_raw",
        "source": str(source),
        "output": str(output),
        "protocol_lock_sha256": hash_text(protocol_path),
        "human_review_decisions_sha256": hash_text(decisions_path),
        "model_blind_build": True,
        "prohibited_signals_used": [],
        "source_images": len(records) + len(exclusions),
        "retained_images": len(records),
        "excluded_images": len(exclusions),
        "exclusion_counts": dict(Counter(row["reason"] for row in exclusions)),
        "reviewed_box_groups": len(cleaning_rows),
        "removed_duplicate_box_rows": sum(
            len(str(row["removed_lines"]).split()) for row in cleaning_rows if row["removed_lines"]
        ),
        "retained_empty_negative_images": sum(not record.label_lines for record in records),
        "classes": dict(enumerate(CLASS_NAMES)),
        "maturity_order": ["young", "green", "half", "red"],
        "maturity_targets_by_class_id": [1.0, 2.0, 3.0, 0.0],
        "policy": {
            "box_iou_review_threshold": BOX_IOU_THRESHOLD,
            "near_phash_hamming_max": NEAR_PHASH_MAX,
            "near_gray32_mae_max": NEAR_GRAY32_MAE_MAX,
            "sequential_gap_max": SEQUENTIAL_GAP_MAX,
            "sequential_gray32_mae_max": SEQUENTIAL_GRAY32_MAE_MAX,
            "split_ratios": SPLIT_RATIOS,
            "seed": SPLIT_SEED,
            "trials": SPLIT_TRIALS,
        },
        "group_count": len(groups),
        "multi_image_group_count": sum(group.image_count > 1 for group in groups),
        "largest_group_sizes": sorted((group.image_count for group in groups), reverse=True)[:10],
        "constraint_edge_counts": dict(Counter(edge["edge_type"] for edge in edges)),
        "split_stats": {
            split: {
                "images": int(values[0]),
                "objects": int(values[1]),
                **{CLASS_NAMES[c]: int(values[2 + c]) for c in range(len(CLASS_NAMES))},
            }
            for split, values in split_stats.items()
        },
        "objective_score": objective,
        "verification": verification,
        "scope_note": (
            "Leakage-repaired same-source group-aware split; not an independent orchard/camera/domain holdout."
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    yaml_path = materialize(
        output, review, records, exclusions, cleaning_rows, groups, edges, assignment, summary
    )
    print(f"[CREATED] {output}")
    print(f"[YAML] {yaml_path}")


if __name__ == "__main__":
    main()
