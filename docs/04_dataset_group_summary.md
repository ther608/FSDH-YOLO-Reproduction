# 04. Dataset groups, names and sizes

This is the quick-reference companion to `docs/03_dataset_protocol.md`.

## 1. Dataset names and classes

| Item | Dataset A | Dataset B |
|---|---|---|
| Manuscript name | Orchard litchi maturity dataset | lichi-maturity (cross-dataset) |
| Repo shorthand | `dataset_a` | `dataset_b` |
| Subset directory | `lychee_groupaware_clean_aug_v2` | `lichi_701515_groupaware_clean_v4_aug` |
| Public source | Zhang et al., arXiv:2510.16800 (RGB-D lychee) | Zhao, Mendeley Data V1, doi:10.17632/c3rk9gv4w9.1 |
| Original images | 878 (RGB) | 1,649 |
| After cleaning | 842 outdoor orchard | 1,495 |
| Classes | 3 | 4 (native, not merged) |
| Class names by ID | `0 unripe`, `1 semi-ripe`, `2 ripe` | `0 green`, `1 half`, `2 red`, `3 young` |
| Maturity order | unripe -> semi-ripe -> ripe | young -> green -> half -> red (ID order is different; young=3) |
| Image scale | about 1280x1024 | 640x640 |
| Paper use | Sections 3.3-3.7 | Section 3.8 cross-dataset |

> Dataset B class IDs are not ordinal. The name order `green, half, red, young`
> is not the ID order or the maturity order.

## 2. Fixed split totals

### Dataset A

| split | Groups | Images | GT | unripe | semi-ripe | ripe |
|---|---:|---:|---:|---:|---:|---:|
| train | 553 | 590 | 11,083 | 3,835 | 2,601 | 4,647 |
| val | 58 | 126 | 2,690 | 909 | 657 | 1,124 |
| test | 61 | 126 | 2,775 | 957 | 653 | 1,165 |
| **total** | **672** | **842** | **16,548** | **5,701** | **3,911** | **6,936** |

### Dataset B

| split | Groups | Images | GT | green | half | red | young |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 949 | 1,071 | 12,935 | 3,839 | 765 | 5,288 | 3,043 |
| val | 118 | 212 | 3,267 | 970 | 191 | 1,337 | 769 |
| test | 121 | 212 | 3,263 | 970 | 187 | 1,337 | 769 |
| **total** | **1,188** | **1,495** | **19,465** | **5,779** | **1,143** | **7,962** | **4,581** |

Both splits are group-level; `group_manifest.csv` row counts equal the totals.
No group crosses splits (`build_summary.json` verification).

## 3. Dataset A: batches, groups and images

### 3.1 Acquisition batches

| Batch | Meaning | Images | Groups |
|---|---|---:|---:|
| `250605_Outdoor_sunny` | 2025-06-05 outdoor sunny | 203 | 195 |
| `250611_Outdoor_rain` | 2025-06-11 outdoor rain | 166 | 104 |
| `250612_Outdoor_sunny` | 2025-06-12 outdoor sunny | 200 | 114 |
| `250619_Outdoor_sunny` | 2025-06-19 outdoor sunny | 273 | 259 |
| **total** | | **842** | **672** |

Group counts use the group's main batch; image counts use `split_manifest.csv`.

### 3.2 Batch x split (images)

| Batch | train | val | test | total |
|---|---:|---:|---:|---:|
| 250605_Outdoor_sunny | 164 | 22 | 17 | 203 |
| 250611_Outdoor_rain | 119 | 15 | 32 | 166 |
| 250612_Outdoor_sunny | 94 | 54 | 52 | 200 |
| 250619_Outdoor_sunny | 213 | 35 | 25 | 273 |
| **total** | **590** | **126** | **126** | **842** |

### 3.3 Group-size distribution

| Images per group | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 9 | 12 | 14 | 26 | 28 | 38 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Groups | 640 | 15 | 5 | 3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

- Single-image groups: 640
- Multi-image groups: **32**
- Largest group: **38**
- Total groups: **672**

### 3.4 Group IDs and filenames

- Group IDs: `G0001` ... `G0672` in `group_manifest.csv`.
- Filenames keep the original batch prefix and class counts, e.g.
  `250605_Outdoor_sunny_ID-0001_unripe-0_semi-ripe-0_ripe-3_sim-0.0016.jpg`.
- Each file in `split_manifest.csv` has `group_id`, `new_split`, `batch`,
  `image_id`, `filename`, `class_0..2`, `image_sha256`.

## 4. Dataset B: sequence families, groups and images

### 4.1 Sequence families

| `sequence_family` | Description | Images | Groups |
|---|---|---:|---:|
| `lychee` | main lychee-prefixed sequence | 846 | 814 |
| `dash` | dash/negative-prefix sequence | 259 | 13 |
| `img_20230519` | IMG_20230519_* acquisition day | 251 | 240 |
| `img_20230621` | IMG_20230621_* acquisition day | 113 | 105 |
| `img_20230613` | IMG_20230613_* acquisition day | 26 | 16 |
| **total** | | **1,495** | **1,188** |

`group_manifest.csv` does not contain `sequence_family`; the field is in
`split_manifest.csv`. A group normally belongs to one family.

### 4.2 Group-size distribution

| Images per group | 1 | 2 | 3 | 4 | 5 | 6 | 54 | 66 | 123 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Groups | 1,139 | 34 | 8 | 1 | 1 | 2 | 1 | 1 | 1 |

- Single-image groups: 1,139
- Multi-image groups: **49**
- Largest group: **123**
- Total groups: **1,188**

### 4.3 Group IDs and filenames

- Group IDs: `G0001` ... `G1188`.
- Original Roboflow-style filenames are preserved, e.g.
  `-1-_jpg.rf.acf0acc89ce8b077384fb4d7773d0982.jpg`.
- `split_manifest.csv` has `group_id`, `new_split`, `legacy_split`, `filename`,
  `objects`, `class_0..3`, `image_sha256`, `sequence_family`, `sequence_value`.
- Ten duplicate-box decisions are in `label_cleaning_manifest.csv`; 154 removed
  flip/rotation images are in `excluded_images.csv`.

## 5. Name summary

### Datasets

- Dataset A: **Orchard litchi maturity dataset**, subset
  **`lychee_groupaware_clean_aug_v2`**, source **An RGB-D image dataset for
  lychee detection and maturity classification for robotic harvesting**
  (Zhang et al., 2025).
- Dataset B: **lichi-maturity**, subset
  **`lichi_701515_groupaware_clean_v4_aug`**, source **lichi-maturity**
  (Zhao, 2025, Mendeley Data V1).

### Classes

- A: `unripe`, `semi-ripe`, `ripe` (IDs 0/1/2).
- B: `green`, `half`, `red`, `young` (IDs 0/1/2/3); maturity order
  `young -> green -> half -> red`.

### Batches / sequence families

- A batches: `250605_Outdoor_sunny`, `250611_Outdoor_rain`,
  `250612_Outdoor_sunny`, `250619_Outdoor_sunny`.
- B families: `lychee`, `dash`, `img_20230519`, `img_20230621`, `img_20230613`.

### Splits and groups

- Split names: `train`, `val`, `test` (fixed; never re-randomized).
- Group IDs: A `G0001`-`G0672`; B `G0001`-`G1188`.
- Manifest names:
  - A: `build_summary.json`, `group_manifest.csv`, `split_manifest.csv`,
    `group_constraint_edges.csv`, `augmentation_log.csv`,
    `augmentation_scheme.txt`;
  - B: `build_summary.json`, `raw_build_summary.json`, `group_manifest.csv`,
    `split_manifest.csv`, `image_constraint_edges.csv`,
    `label_cleaning_manifest.csv`, `excluded_images.csv`,
    `augmentation_summary.json`, `augmentation_log.csv`,
    `audit/{protocol_lock,review_decisions,box_review_manifest}.json`,
    `dataset_b_v4_verification.json`.

## 6. Consistency checks

| Check | A | B |
|---|---|---|
| Groups = group_manifest rows | 672 | 1,188 |
| Images = split_manifest rows | 842 | 1,495 |
| Split images = sum of group image_count | 590/126/126 | 1,071/212/212 |
| Split GT = sum of group object_count | 11,083/2,690/2,775 | 12,935/3,267/3,263 |
| Group cross-split violations | 0 | 0 |
| Constraint-edge cross-split violations | 0 / 584 | 0 / 730 |
| Exact-hash cross-split | 0/0/0 | 0/0/0 |
| Augmented train images | 2,360 | 4,284 |
| Val/test augmented? | No | No (byte-identical) |