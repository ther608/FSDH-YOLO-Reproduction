# Dataset B fixed split / cleaning / grouping manifests

Source: Zhao, *lichi-maturity*, Mendeley Data V1,
`doi:10.17632/c3rk9gv4w9.1`. Images are not redistributed; this directory keeps
the fixed split, cleaning/grouping manifests, audit and augmentation records.

## Files

| File | Content |
|---|---|
| `build_summary.json` | 1,649 source / 1,495 retained / 1,188 groups / 730 edges / split stats / zero violations |
| `raw_build_summary.json` | Raw build summary stored in the augmented directory |
| `group_manifest.csv` | 1,188 groups: `group_id, split, image_count, object_count, class_0..3` |
| `split_manifest.csv` | 1,495 images: `group_id, new_split, legacy_split, filename, objects, class_0..3, image_sha256, sequence_family, sequence_value` |
| `image_constraint_edges.csv` | 730 edges (sequential 723 + near 7) |
| `label_cleaning_manifest.csv` | 10 duplicate-box decisions |
| `excluded_images.csv` | 154 removed source-side flip/rotation images |
| `audit/` | `protocol_lock.json`, `review_decisions.json`, `box_review_manifest.json` |
| `dataset_b_v4_verification.json` | Independent verification PASS |
| `augmentation_summary.json` | 4,284 images / 51,728 GT / class and operation counts |
| `augmentation_log.csv` | 3,213 records (1,071 x 3 variants) |

## Fixed numbers

| Item | Value |
|---|---|
| Original images | 1,649 |
| Removed flip/rotation images | 154 |
| Removed duplicate boxes | 10 |
| Retained images | 1,495 (including 2 empty-label negatives) |
| Groups / multi-image / largest | 1,188 / 49 / 123 |
| Split images | 1,071 / 212 / 212 |
| Split groups | 949 / 118 / 121 |
| Split GT | 12,935 / 3,267 / 3,263 |
| Augmented train | 4,284 images / 51,728 GT |
| Classes | 0 green / 1 half / 2 red / 3 young |
| Maturity order | young -> green -> half -> red |
| Split seed | 20260731 |

> `young=3`: class IDs are not maturity order. Use `maturity_order` /
> `maturity_targets_by_class_id` for ordinal work.

## Restoration

Find each source image by `filename`, verify `image_sha256`, and copy it into
`train/val/test` `images`/`labels` according to `new_split`. Generate three
training variants from `augmentation_log.csv`; leave val/test byte-identical.

Full rules: `docs/03_dataset_protocol.md`. Sequence-family summary:
`docs/04_dataset_group_summary.md`.