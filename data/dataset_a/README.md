# Dataset A fixed split / cleaning / grouping manifests

Source: Zhang et al., *An RGB-D image dataset for lychee detection and maturity
classification for robotic harvesting*, arXiv:2510.16800. This work uses the
**RGB subset** only (no depth/IR/LiDAR).

**Images are not in this repository.** This directory keeps the fixed split,
group assignment, constraint edges and augmentation log.

## Files

| File | Content |
|---|---|
| `build_summary.json` | 842 images / 672 groups / 32 multi-image groups / 584 edges / split stats / zero violations |
| `group_manifest.csv` | 672 groups: `group_id, split, image_count, object_count, class_0..2, batch_counts` |
| `split_manifest.csv` | 842 images: `group_id, new_split, old_split, batch, image_id, filename, class_0..2, image_sha256` |
| `group_constraint_edges.csv` | 584 grouping edges (near / sequential) |
| `augmentation_log.csv` | 1,770 training augmentation records (590 x 3 variants, with per-image parameters) |
| `augmentation_scheme.txt` | Three augmentation categories and parameter ranges |

## Fixed numbers

| Item | Value |
|---|---|
| Original images | 878 |
| Indoor tabletop removed | 36 |
| Outdoor retained | 842 |
| Duplicate-annotation candidate groups / boxes | 936 / 1,873 |
| Duplicate boxes removed | 933 (unripe 56 / semi-ripe 554 / ripe 323) |
| Groups / multi-image / largest | 672 / 32 / 38 |
| Split images | 590 / 126 / 126 |
| Split groups | 553 / 58 / 61 |
| Split GT | 11,083 / 2,690 / 2,775 |
| Augmented train | 2,360 images / 44,151 GT |
| Split seed | 20260731 |

## Restoration

Find each source image by `filename`, verify `image_sha256`, and copy it into
`images/{train,val,test}` and `labels/{train,val,test}` according to
`new_split`. Generate three augmented variants for training images from
`augmentation_log.csv`; leave val/test unchanged.

Full rules: `docs/03_dataset_protocol.md`. Group/batch summary:
`docs/04_dataset_group_summary.md`.