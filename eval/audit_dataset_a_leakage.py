"""Audit Dataset A provenance and cross-split image leakage.

This script is read-only with respect to the dataset. It reconciles the 878
images documented by the dataset authors with the 842 outdoor images used by
the paper, verifies that every selected source image occurs exactly once in the
current split, and audits exact, perceptual, and sequential-neighbour leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")
BATCH_RE = re.compile(r"Image_YoloLabel_(\d{6}_(?:Indoor|Outdoor)_[A-Za-z]+)")
SPLIT_RE = re.compile(r"^(\d{6}_(?:Indoor|Outdoor)_[A-Za-z]+)_(ID-(\d+)_.*)$")
SIM_RE = re.compile(r"_sim-([0-9.]+)$")


@dataclass
class Record:
    batch: str
    image_id: int
    source_image: Path
    source_label: Path
    split: str
    split_image: Path
    split_label: Path
    sha256: str
    label_sha256: str
    phash: int
    gray32: np.ndarray
    filename_sim: float | None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_signature(path: Path) -> tuple[int, np.ndarray]:
    with Image.open(path) as image:
        gray = np.asarray(
            image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
            dtype=np.float32,
        ) / 255.0
    n = 32
    k = np.arange(n, dtype=np.float32)[:, None]
    x = np.arange(n, dtype=np.float32)[None, :]
    basis = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    dct = basis @ gray @ basis.T
    low = dct[:8, :8].reshape(-1)[1:]
    median = float(np.median(low))
    value = 0
    for bit in low > median:
        value = (value << 1) | int(bit)
    return value, gray


def source_batches(dataset_root: Path) -> dict[str, tuple[Path, Path]]:
    batches: dict[str, tuple[Path, Path]] = {}
    for outer in sorted(dataset_root.glob("Image_YoloLabel_*")):
        if not outer.is_dir():
            continue
        match = BATCH_RE.search(outer.name)
        if not match:
            continue
        inner = outer / outer.name
        image_dir = inner / "images" / "raw" / "full_integration"
        label_dir = inner / "labels" / "raw" / "full_integration"
        if image_dir.is_dir() and label_dir.is_dir():
            batches[match.group(1)] = (image_dir, label_dir)
    return batches


def parse_source_id(path: Path) -> int:
    match = re.match(r"ID-(\d+)_", path.stem)
    if not match:
        raise ValueError(f"Cannot parse source ID from {path.name}")
    return int(match.group(1))


def parse_filename_sim(path: Path) -> float | None:
    match = SIM_RE.search(path.stem)
    return float(match.group(1)) if match else None


def load_records(dataset_root: Path) -> tuple[list[Record], list[dict[str, object]], list[str]]:
    raw_root = dataset_root / "divide_stratified_raw"
    batches = source_batches(dataset_root)
    source_by_key: dict[tuple[str, str], tuple[Path, Path]] = {}
    source_counts: Counter[str] = Counter()
    errors: list[str] = []

    for batch, (image_dir, label_dir) in batches.items():
        for image in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            label = label_dir / f"{image.stem}.txt"
            key = (batch, image.name)
            if key in source_by_key:
                errors.append(f"Duplicate source key: {batch}/{image.name}")
            if not label.is_file():
                errors.append(f"Missing source label: {label}")
            source_by_key[key] = (image, label)
            source_counts[batch] += 1

    split_by_key: dict[tuple[str, str], tuple[str, Path, Path]] = {}
    split_counts: Counter[str] = Counter()
    for split in SPLITS:
        image_dir = raw_root / "images" / split
        label_dir = raw_root / "labels" / split
        for image in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            match = SPLIT_RE.match(image.name)
            if not match:
                errors.append(f"Cannot parse split filename: {image}")
                continue
            batch, source_name = match.group(1), match.group(2)
            key = (batch, source_name)
            label = label_dir / f"{image.stem}.txt"
            if key in split_by_key:
                errors.append(f"Source appears more than once in split: {batch}/{source_name}")
            if not label.is_file():
                errors.append(f"Missing split label: {label}")
            split_by_key[key] = (split, image, label)
            split_counts[split] += 1

    missing_from_split = sorted(set(source_by_key) - set(split_by_key))
    unknown_in_split = sorted(set(split_by_key) - set(source_by_key))
    errors.extend(f"Source missing from split: {batch}/{name}" for batch, name in missing_from_split)
    errors.extend(f"Split image not found in sources: {batch}/{name}" for batch, name in unknown_in_split)

    records: list[Record] = []
    trace: list[dict[str, object]] = []
    for key in sorted(set(source_by_key) & set(split_by_key)):
        batch, source_name = key
        source_image, source_label = source_by_key[key]
        split, split_image, split_label = split_by_key[key]
        source_image_hash = sha256(source_image)
        split_image_hash = sha256(split_image)
        source_label_hash = sha256(source_label)
        split_label_hash = sha256(split_label)
        phash, gray32 = image_signature(source_image)
        record = Record(
            batch=batch,
            image_id=parse_source_id(source_image),
            source_image=source_image,
            source_label=source_label,
            split=split,
            split_image=split_image,
            split_label=split_label,
            sha256=source_image_hash,
            label_sha256=source_label_hash,
            phash=phash,
            gray32=gray32,
            filename_sim=parse_filename_sim(source_image),
        )
        records.append(record)
        trace.append(
            {
                "batch": batch,
                "image_id": record.image_id,
                "source_filename": source_name,
                "split": split,
                "split_filename": split_image.name,
                "source_image_sha256": source_image_hash,
                "split_image_sha256": split_image_hash,
                "image_bytes_identical": source_image_hash == split_image_hash,
                "source_label_sha256": source_label_hash,
                "split_label_sha256": split_label_hash,
                "label_bytes_identical": source_label_hash == split_label_hash,
                "filename_sim": record.filename_sim,
                "source_image_path": str(source_image),
                "split_image_path": str(split_image),
            }
        )

    reconciliation = []
    documented = {
        "250605_Outdoor_sunny": 203,
        "250610_Indoor_table": 36,
        "250611_Outdoor_rain": 166,
        "250612_Outdoor_sunny": 200,
        "250619_Outdoor_sunny": 273,
    }
    for batch, expected in documented.items():
        observed = source_counts[batch]
        selected = sum(record.batch == batch for record in records)
        reconciliation.append(
            {
                "batch": batch,
                "documented_original_images": expected,
                "locally_observed_source_images": observed,
                "selected_into_dataset_a": selected,
                "status": (
                    "selected_outdoor_batch"
                    if selected == expected and "Outdoor" in batch
                    else "documented_indoor_batch_excluded_and_not_present_locally"
                    if batch == "250610_Indoor_table" and observed == 0 and selected == 0
                    else "mismatch"
                ),
            }
        )
    if sum(documented.values()) != 878:
        errors.append("Internal documented count does not sum to 878")
    if sum(source_counts.values()) != 842:
        errors.append(f"Observed outdoor source count is {sum(source_counts.values())}, expected 842")

    return records, reconciliation, errors


def duplicate_audit(
    records: list[Record],
    phash_limit: int,
    mae_limit: float,
    cross_split_only: bool = True,
) -> tuple[list[dict], list[dict]]:
    exact_groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        exact_groups[record.sha256].append(record)
    exact_rows = []
    for digest, group in exact_groups.items():
        if len(group) <= 1:
            continue
        exact_rows.append(
            {
                "sha256": digest,
                "count": len(group),
                "splits": "|".join(sorted({item.split for item in group})),
                "images": "|".join(str(item.split_image) for item in group),
            }
        )

    near_rows = []
    for left in range(len(records)):
        a = records[left]
        for right in range(left + 1, len(records)):
            b = records[right]
            if cross_split_only and a.split == b.split:
                continue
            hamming = (a.phash ^ b.phash).bit_count()
            if hamming > phash_limit:
                continue
            mae = float(np.abs(a.gray32 - b.gray32).mean())
            if mae > mae_limit:
                continue
            near_rows.append(
                {
                    "batch_a": a.batch,
                    "id_a": a.image_id,
                    "split_a": a.split,
                    "image_a": a.split_image.name,
                    "batch_b": b.batch,
                    "id_b": b.image_id,
                    "split_b": b.split,
                    "image_b": b.split_image.name,
                    "cross_split": a.split != b.split,
                    "same_batch": a.batch == b.batch,
                    "id_gap": abs(a.image_id - b.image_id) if a.batch == b.batch else "",
                    "phash_hamming": hamming,
                    "gray32_mae": round(mae, 6),
                    "high_confidence": hamming <= 6 and mae <= 0.035,
                }
            )
    near_rows.sort(key=lambda row: (row["phash_hamming"], row["gray32_mae"]))
    return exact_rows, near_rows


def sequential_audit(
    records: list[Record], max_gap: int, cross_split_only: bool = True
) -> list[dict]:
    by_batch: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_batch[record.batch].append(record)
    rows = []
    for batch, group in by_batch.items():
        group.sort(key=lambda record: record.image_id)
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                gap = group[right].image_id - group[left].image_id
                if gap > max_gap:
                    break
                if cross_split_only and group[left].split == group[right].split:
                    continue
                hamming = (group[left].phash ^ group[right].phash).bit_count()
                mae = float(np.abs(group[left].gray32 - group[right].gray32).mean())
                rows.append(
                    {
                        "batch": batch,
                        "id_a": group[left].image_id,
                        "split_a": group[left].split,
                        "image_a": group[left].split_image.name,
                        "id_b": group[right].image_id,
                        "split_b": group[right].split,
                        "image_b": group[right].split_image.name,
                        "cross_split": group[left].split != group[right].split,
                        "id_gap": gap,
                        "phash_hamming": hamming,
                        "gray32_mae": round(mae, 6),
                        "filename_sim_a": group[left].filename_sim,
                        "filename_sim_b": group[right].filename_sim,
                    }
                )
    rows.sort(key=lambda row: (row["id_gap"], row["phash_hamming"], row["gray32_mae"]))
    return rows


def high_confidence_groups(
    records: list[Record], near_rows: list[dict], require_cross_split: bool = True
) -> list[dict]:
    parent = list(range(len(records)))
    by_name = {record.split_image.name: index for index, record in enumerate(records)}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for row in near_rows:
        if row["high_confidence"]:
            union(by_name[row["image_a"]], by_name[row["image_b"]])

    groups: dict[int, list[Record]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[find(index)].append(record)

    rows = []
    group_id = 0
    for members in groups.values():
        splits = sorted({member.split for member in members})
        if len(members) <= 1 or (require_cross_split and len(splits) <= 1):
            continue
        group_id += 1
        rows.append(
            {
                "group_id": group_id,
                "count": len(members),
                "splits": "|".join(splits),
                "batches": "|".join(sorted({member.batch for member in members})),
                "members": "|".join(
                    f"{member.split}:{member.split_image.name}" for member in sorted(
                        members, key=lambda item: (item.split, item.batch, item.image_id)
                    )
                ),
            }
        )
    rows.sort(key=lambda row: (-row["count"], row["group_id"]))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phash-limit", type=int, default=10)
    parser.add_argument("--mae-limit", type=float, default=0.06)
    parser.add_argument("--sequential-gap", type=int, default=3)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    records, reconciliation, errors = load_records(args.dataset_root)
    exact_rows, all_near_rows = duplicate_audit(
        records, args.phash_limit, args.mae_limit, cross_split_only=False
    )
    near_rows = [row for row in all_near_rows if row["cross_split"]]
    high_confidence_group_rows = high_confidence_groups(records, near_rows)
    all_high_confidence_group_rows = high_confidence_groups(
        records, all_near_rows, require_cross_split=False
    )
    all_sequential_rows = sequential_audit(
        records, args.sequential_gap, cross_split_only=False
    )
    sequential_rows = [row for row in all_sequential_rows if row["cross_split"]]

    trace_rows = []
    for record in records:
        source_image_hash = sha256(record.source_image)
        split_image_hash = sha256(record.split_image)
        source_label_hash = sha256(record.source_label)
        split_label_hash = sha256(record.split_label)
        trace_rows.append(
            {
                "batch": record.batch,
                "image_id": record.image_id,
                "source_filename": record.source_image.name,
                "split": record.split,
                "split_filename": record.split_image.name,
                "image_bytes_identical": source_image_hash == split_image_hash,
                "label_bytes_identical": source_label_hash == split_label_hash,
                "sha256": source_image_hash,
                "label_sha256": source_label_hash,
                "filename_sim": record.filename_sim,
            }
        )

    write_csv(args.out / "population_reconciliation.csv", reconciliation)
    write_csv(args.out / "dataset_a_trace_manifest.csv", trace_rows)
    write_csv(args.out / "exact_duplicate_groups.csv", exact_rows)
    write_csv(args.out / "near_duplicate_cross_split.csv", near_rows)
    write_csv(args.out / "near_duplicate_all_pairs.csv", all_near_rows)
    write_csv(args.out / "high_confidence_cross_split_groups.csv", high_confidence_group_rows)
    write_csv(args.out / "high_confidence_all_groups.csv", all_high_confidence_group_rows)
    write_csv(args.out / "sequential_cross_split_pairs.csv", sequential_rows)
    write_csv(args.out / "sequential_all_pairs.csv", all_sequential_rows)

    split_counts = Counter(record.split for record in records)
    batch_splits: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        batch_splits[record.batch][record.split] += 1
    high_confidence_names = {
        name
        for row in near_rows
        if row["high_confidence"]
        for name in (row["image_a"], row["image_b"])
    }
    affected_by_split = Counter(
        record.split for record in records if record.split_image.name in high_confidence_names
    )
    high_confidence_pair_count = sum(bool(row["high_confidence"]) for row in near_rows)
    summary = {
        "status": (
            "FAIL_PROVENANCE"
            if errors
            else "FAIL_STRICT_SPLIT_INDEPENDENCE"
            if high_confidence_pair_count or all(
                all(counts[split] > 0 for split in SPLITS) for counts in batch_splits.values()
            )
            else "PASS"
        ),
        "provenance_status": "PASS" if not errors else "FAIL",
        "split_independence_status": (
            "FAIL_HIGH_RISK" if high_confidence_pair_count else "WARN_SHARED_ACQUISITION_BATCHES"
        ),
        "dataset_root": str(args.dataset_root),
        "documented_population": 878,
        "documented_indoor_exclusion": 36,
        "selected_outdoor_population": len(records),
        "split_counts": dict(split_counts),
        "batch_split_counts": {batch: dict(counts) for batch, counts in sorted(batch_splits.items())},
        "all_selected_images_byte_identical_to_source": all(
            sha256(record.source_image) == sha256(record.split_image) for record in records
        ),
        "all_selected_labels_byte_identical_to_source": all(
            sha256(record.source_label) == sha256(record.split_label) for record in records
        ),
        "exact_duplicate_groups": len(exact_rows),
        "cross_split_exact_duplicate_groups": sum(
            len(row["splits"].split("|")) > 1 for row in exact_rows
        ),
        "cross_split_near_duplicate_pairs": len(near_rows),
        "all_near_duplicate_pairs": len(all_near_rows),
        "cross_split_high_confidence_near_duplicate_pairs": high_confidence_pair_count,
        "high_confidence_cross_split_groups": len(high_confidence_group_rows),
        "high_confidence_affected_unique_images": len(high_confidence_names),
        "high_confidence_affected_unique_images_by_split": dict(affected_by_split),
        "cross_split_sequential_pairs_gap_le_3": len(sequential_rows),
        "all_sequential_pairs_gap_le_3": len(all_sequential_rows),
        "cross_split_adjacent_id_pairs": sum(row["id_gap"] == 1 for row in sequential_rows),
        "all_acquisition_batches_present_in_all_splits": all(
            all(counts[split] > 0 for split in SPLITS) for counts in batch_splits.values()
        ),
        "thresholds": {
            "near_candidate_phash_hamming_max": args.phash_limit,
            "near_candidate_gray32_mae_max": args.mae_limit,
            "high_confidence_phash_hamming_max": 6,
            "high_confidence_gray32_mae_max": 0.035,
            "sequential_id_gap_max": args.sequential_gap,
        },
        "errors": errors,
    }
    (args.out / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
