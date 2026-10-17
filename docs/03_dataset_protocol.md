# 03. Dataset protocols: fixed splits, cleaning and grouping

This document aligns the manuscript's Appendix A.1 text with the machine-readable
manifests in `data/`. **Authority order**: `data/*/build_summary.json` and
`data/*/*manifest*.csv` > manuscript Appendix A > this note.

---

## 1. Dataset A: outdoor orchard, three maturity classes

### 1.1 Source and image filtering

| Item | Value |
|---|---|
| Source | Zhang et al., *An RGB-D image dataset for lychee detection and maturity classification for robotic harvesting*, arXiv:2510.16800 |
| Used part | RGB subset only (no depth/IR/LiDAR) |
| Original images | 878 |
| Removed | 36 indoor tabletop images |
| Retained | 842 outdoor natural-orchard images |
| Acquisition batches | `250605_Outdoor_sunny`, `250611_Outdoor_rain`, `250612_Outdoor_sunny`, `250619_Outdoor_sunny` |
| Classes | `0 unripe`, `1 semi-ripe`, `2 ripe` |

### 1.2 Annotation cleaning (duplicate boxes in the same image)

1. Any two boxes in the same image with `IoU > 0.90` entered manual review.
2. This produced **936 candidate groups** involving **1,873 annotated boxes**.
3. Manual review:
   - **932 groups** kept one box (the correct localization/category);
   - **4 groups** kept all overlapping boxes (confirmed different fruits).
4. **933 duplicate boxes** were removed:
   - unripe: **56**
   - semi-ripe: **554**
   - ripe: **323**
5. No same-class conflict pairs were found; all candidates were cross-maturity
   high-overlap cases.

> The 936/1,873 figures describe candidate groups/boxes, not the number removed.
> The removed count is 933.

### 1.3 Grouping (connected components)

Two images are linked if either rule holds:

| Rule | Threshold |
|---|---|
| Visual near-duplicate | 64-bit pHash Hamming distance **<= 10** and 32x32 grayscale MAE **<= 0.06** |
| Same-batch short sequence | Same acquisition batch, image-index difference **<= 3** and 32x32 grayscale MAE **<= 0.10** |

The transitive closure gives the indivisible groups:

| Item | Value |
|---|---|
| Images | 842 |
| Groups | **672** |
| Multi-image groups | **32** |
| Largest group | **38 images** |
| Top 10 group sizes | 38, 28, 26, 14, 12, 9, 7, 6, 5, 4 |
| Constraint edges | **584** (near 349 + sequential 235) |
| Cross-split group violations | **0** |
| Cross-split constraint-edge violations | **0** |
| Exact-hash cross-split | train-val 0 / train-test 0 / val-test 0 |

### 1.4 Fixed split

- Split unit: **group**, not image; a group cannot cross train/val/test.
- Ratio: **70% / 15% / 15%**.
- Build seed: **20260731** (dataset split seed; independent of training seeds 0/1/2).

| split | Groups | Images | GT instances | unripe | semi-ripe | ripe |
|---|---:|---:|---:|---:|---:|---:|
| train | 553 | 590 | 11,083 | 3,835 | 2,601 | 4,647 |
| val | 58 | 126 | 2,690 | 909 | 657 | 1,124 |
| test | 61 | 126 | 2,775 | 957 | 653 | 1,165 |
| **total** | **672** | **842** | **16,548** | **5,701** | **3,911** | **6,936** |

Batch x split (images):

| Batch | train | val | test | total |
|---|---:|---:|---:|---:|
| 250605_Outdoor_sunny | 164 | 22 | 17 | 203 |
| 250611_Outdoor_rain | 119 | 15 | 32 | 166 |
| 250612_Outdoor_sunny | 94 | 54 | 52 | 200 |
| 250619_Outdoor_sunny | 213 | 35 | 25 | 273 |
| **total** | **590** | **126** | **126** | **842** |

### 1.5 Training-only offline augmentation

Each original training image generates **three independent variants**, one
operation from each of three categories. No chained augmentation; no vertical
flip; val/test are copied unchanged.

| Category | Operations | Parameter range |
|---|---|---|
| Illumination/colour | brightness-contrast; HSV | contrast gain [0.88, 1.12]; brightness bias [-18, 18]; saturation scale [0.86, 1.14]; hue shift [-4, 4] |
| Imaging quality | Gaussian blur; sensor noise | 3x3 kernel, blur intensity [0.35, 0.85]; Gaussian noise sigma [3.0, 7.0] |
| Mild geometry | horizontal flip; rotation; translation; scaling | rotation +/-5 deg; translation +/-2%; scale [1.04, 1.10] |

| Item | Raw train | Augmented train |
|---|---:|---:|
| Images | 590 | **2,360** (590 + 590x3) |
| GT instances | 11,083 | **44,151** |
| unripe | 3,835 | **15,268** |
| semi-ripe | 2,601 | **10,372** |
| ripe | 4,647 | **18,511** |

Operation counts (`augmentation_log.csv`, 1,770 rows = 590x3):

| Operation | Count |
|---|---:|
| brightness_contrast | 305 |
| hsv_colour | 285 |
| gaussian_blur | 304 |
| sensor_noise | 286 |
| horizontal_flip | 156 |
| rotation | 143 |
| translation | 154 |
| scale | 137 |
| **total** | **1,770** |

Val/test remain 126 images / 2,690 and 2,775 GT, no augmentation.

### 1.6 Files and YAML

```text
data/dataset_a/
|-- build_summary.json           # 842/672/32/584/split stats/0 violations
|-- group_manifest.csv           # 672 groups
|-- split_manifest.csv           # 842 images: group_id, new_split, batch, image_id, filename, class_0..2, sha256
|-- group_constraint_edges.csv   # 584 edges
|-- augmentation_log.csv         # 1,770 rows with per-image random parameters
`-- augmentation_scheme.txt
```

```yaml
nc: 3
names:
  0: unripe
  1: semi-ripe
  2: ripe
```

---

## 2. Dataset B: `lichi-maturity`, four native classes

### 2.1 Source and image filtering

| Item | Value |
|---|---|
| Source | Zhao, *lichi-maturity*, Mendeley Data V1, `doi:10.17632/c3rk9gv4w9.1` |
| Original images | 1,649 |
| Removed | 154 source-side offline flip/rotation augmented images (identified by filename) |
| Duplicate boxes | 10 candidate groups, 10 duplicate rows removed after manual review |
| Retained | **1,495 images** |
| Empty-label negatives | **2** retained (val) |
| Classes | `0 green`, `1 half`, `2 red`, `3 young` (native, not merged/remapped) |

> **Class IDs are not maturity order.** The maturity order is
> `young -> green -> half -> red`, but `young=3`. Use `maturity_order` /
> `maturity_targets_by_class_id` for ordinal work, not class-ID sorting.

### 2.2 Grouping

Same thresholds as Dataset A:

| Rule | Threshold |
|---|---|
| Visual near-duplicate | 64-bit pHash Hamming **<= 10** and 32x32 grayscale MAE **<= 0.06** |
| Same-batch short sequence | Index difference **<= 3** and 32x32 grayscale MAE **<= 0.10** |

| Item | Value |
|---|---|
| Retained images | 1,495 |
| Groups | **1,188** |
| Multi-image groups | **49** |
| Largest group | **123 images** |
| Top 10 group sizes | 123, 66, 54, 6, 6, 5, 4, 3, 3, 3 |
| Constraint edges | **730** (sequential 723 + near 7) |
| Cross-split group violations | **0** |
| Cross-split constraint-edge violations | **0** |
| Exact-hash cross-split | train-val 0 / train-test 0 / val-test 0 |

### 2.3 Fixed split

- Split unit: **group**; ratio **70/15/15**; build seed **20260731**; optimization trials 1,000.

| split | Groups | Images | GT instances | green | half | red | young |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 949 | 1,071 | 12,935 | 3,839 | 765 | 5,288 | 3,043 |
| val | 118 | 212 | 3,267 | 970 | 191 | 1,337 | 769 |
| test | 121 | 212 | 3,263 | 970 | 187 | 1,337 | 769 |
| **total** | **1,188** | **1,495** | **19,465** | **5,779** | **1,143** | **7,962** | **4,581** |

> The manuscript says 1,071/212/212 and 4,284 augmented; per-split GT totals and
> class counts come from `data/dataset_b/build_summary.json`.

### 2.4 Training-only offline augmentation

Same three-category protocol; no chain; no vertical flip; val/test byte-identical.

| Item | Raw train | Augmented train |
|---|---:|---:|
| Images | 1,071 | **4,284** (1,071 + 1,071x3) |
| GT instances | 12,935 | **51,728** |
| green | 3,839 | **15,355** |
| half | 765 | **3,060** |
| red | 5,288 | **21,145** |
| young | 3,043 | **12,168** |

Operation counts (`augmentation_log.csv`, 3,213 rows = 1,071x3):

| Operation | Count |
|---|---:|
| brightness_contrast | 534 |
| hsv_colour | 537 |
| gaussian_blur | 534 |
| sensor_noise | 537 |
| horizontal_flip | 271 |
| rotation | 273 |
| translation | 271 |
| scale | 256 |
| **total** | **3,213** |

### 2.5 Files and YAML

```text
data/dataset_b/
|-- build_summary.json             # 1,649/1,495/154/1,188/49/730/split stats/0 violations
|-- raw_build_summary.json
|-- group_manifest.csv              # 1,188 groups
|-- split_manifest.csv              # 1,495 images
|-- image_constraint_edges.csv      # 730 edges
|-- label_cleaning_manifest.csv     # 10 duplicate-box decisions
|-- excluded_images.csv             # 154 removed flip/rotation images
|-- audit/                          # protocol_lock.json, review_decisions.json, box_review_manifest.json
|-- dataset_b_v4_verification.json  # independent verification PASS
|-- augmentation_summary.json       # 4,284/51,728/class and operation counts
`-- augmentation_log.csv            # 3,213 rows
```

```yaml
nc: 4
names:
  0: green
  1: half
  2: red
  3: young
```

`maturity_order: [young, green, half, red]`
`maturity_targets_by_class_id: [1.0, 2.0, 3.0, 0.0]`
(i.e. class 0 green -> target 1; 1 half -> 2; 2 red -> 3; 3 young -> 0).
---

## 3. Machine-readable split fields

### 3.1 Dataset A `split_manifest.csv`

| Column | Meaning |
|---|---|
| `group_id` | Group ID (`G0001` ...) |
| `new_split` | Final split: train/val/test |
| `old_split` | Original dataset split (provenance only) |
| `batch` | Acquisition batch |
| `image_id` | Sequence index within batch |
| `filename` | Original filename (contains class counts and sim value) |
| `class_0/1/2` | Image-level unripe/semi-ripe/ripe GT counts |
| `image_sha256` | Pixel-level SHA256 of the source image |

`group_manifest.csv` additionally has `image_count`, `object_count` and
`batch_counts` (JSON).

### 3.2 Dataset B `split_manifest.csv`

| Column | Meaning |
|---|---|
| `group_id` | Group ID |
| `new_split` | Final split |
| `legacy_split` | Original dataset split (provenance only) |
| `filename` | Original filename |
| `objects` | Image-level GT count |
| `class_0..3` | green/half/red/young GT counts |
| `image_sha256` | Pixel-level SHA256 of the source image |
| `sequence_family` | Acquisition sequence family (lychee/dash/img_20230519/...) |
| `sequence_value` | Index within sequence |

### 3.3 Reconstruction steps

1. Download the original public dataset.
2. Find each source image by `filename` and verify `image_sha256`.
3. Copy/hardlink to `images/{train,val,test}` and `labels/{train,val,test}`
   according to `new_split`.
4. **Do not re-randomize**; `group_id` already binds near-duplicate/short-sequence
   images.
5. Generate training-only augmentation using the per-image parameters in
   `augmentation_log.csv`; leave val/test unchanged.
6. Verify counts:
   - A: 590/126/126 images; 11,083/2,690/2,775 GT;
   - B: 1,071/212/212 images; 12,935/3,267/3,263 GT.

---

## 4. Cleaning/grouping scripts and audit

| Dataset | Script | Status |
|---|---|---|
| A | `eval/audit_dataset_a_leakage.py` | Included; read-only audit of 842 vs 878, exact/perceptual/sequential leakage |
| A original build | - | Not preserved; use `build_summary.json` + the three manifests as authority |
| B build | `eval/build_lichi_dataset_b_v4.py` | Included |
| B audit | `eval/audit_lichi_dataset_b_v4.py` | Included |
| B augmentation | `eval/augment_lichi_dataset_b_v4.py` | Included |
| B verification | `eval/verify_lichi_dataset_b_v4.py` | Included; independent verification PASS |

Dataset B `scope_note`:

> "Leakage-repaired same-source group-aware split; not an independent
> orchard/camera/domain holdout."

Thus Dataset B is a same-source, leakage-repaired split, not an independent
orchard/camera/domain test set. Section 3.8 conclusions should respect this.

---

## 5. Notes and discrepancies

1. Dataset A has **936 duplicate-annotation candidate groups / 1,873 boxes**,
   while `build_summary.json` reports **584 image-grouping constraint edges**.
   They are different concepts: 936 is annotation duplicates, 584 is image
   grouping.
2. Dataset B retains 1,495 original images; augmentation expands only the
   training split to 4,284. Val/test remain 212/212 originals.
3. Dataset A test is 126 images, not 212. `AP_small` area definitions differ:
   A uses `[0,4096) px2` (`--area-scale 4`), B uses `<1024 px2`
   (`--area-scale 1`).
4. Dataset B class IDs are not ordinal: `young=3`. The paper's class order
   `green/half/red/young` is a name listing, not ID or maturity order.
5. Split seed **20260731** is separate from training seeds **0/1/2**.