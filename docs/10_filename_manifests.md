# 10. Complete filename manifests and augmentation naming

This repository uploads **complete filename lists, not images**. Both datasets
encode the offline augmentation process in the filename, so the full augmented
dataset can be audited from the manifests alone.

## 1. Files

| Dataset | Manifest | Rows |
|---|---|---:|
| A | `data/dataset_a/filename_manifest.csv` | 2,612 |
| B | `data/dataset_b/filename_manifest.csv` | 4,708 |

A: 590 original train + 1,770 augmented train + 126 val + 126 test = 2,612.
B: 1,071 original train + 3,213 augmented train + 212 val + 212 test = 4,708.

No image files are included.

## 2. Manifest columns

| Column | Meaning |
|---|---|
| `dataset` | `dataset_a` or `dataset_b` |
| `split` | train / val / test |
| `subset` | `original` or `augmented` |
| `filename` | Exact image filename |
| `group_id` | Group ID from `split_manifest.csv`; augmented images inherit the source group |
| `source_image` | Original source image; equals `filename` for originals |
| `aug_variant` | 1 / 2 / 3 for augmented images; empty for originals |
| `aug_category` | `illum_color`, `image_quality`, `mild_geometry`; empty for originals |
| `aug_operation` | Operation name (see below); empty for originals |
| `aug_parameters` | JSON string of the sampled parameters (from `augmentation_log.csv`) |
| `objects` | GT instances in the image |
| `class_0..N` | Per-class GT counts (A: 3 classes; B: 4) |
| `image_sha256` | SHA256 of the image file |

## 3. Dataset A naming pattern

### 3.1 Original images

```text
{batch}_ID-{image_id}_unripe-{n}_semi-ripe-{n}_ripe-{n}_sim-{value}.jpg
```

Example:

```text
250605_Outdoor_sunny_ID-0001_unripe-0_semi-ripe-0_ripe-3_sim-0.0016.jpg
```

- `batch`: `250605_Outdoor_sunny`, `250611_Outdoor_rain`,
  `250612_Outdoor_sunny`, `250619_Outdoor_sunny`.
- `unripe-{n}_semi-ripe-{n}_ripe-{n}` are the per-image class counts.
- `sim-{value}` is the similarity value provided by the original dataset.

### 3.2 Augmented images

The suffix is:

```text
_aug{variant}_{category}_{operation}.jpg
```

| Variant | Category | Operations |
|---|---|---|
| `aug1` | `illum_color` | `brightness_contrast`, `hsv_colour` |
| `aug2` | `image_quality` | `gaussian_blur`, `sensor_noise` |
| `aug3` | `mild_geometry` | `horizontal_flip`, `rotation`, `translation`, `scale` |

Examples:

```text
..._sim-0.0016_aug1_illum_color_brightness_contrast.jpg
..._sim-0.0016_aug2_image_quality_gaussian_blur.jpg
..._sim-0.0016_aug3_mild_geometry_translation.jpg
```

The exact sampled parameters are in `aug_parameters` and in
`data/dataset_a/augmentation_log.csv`.

## 4. Dataset B naming pattern

### 4.1 Original images

Original Roboflow-style names are preserved, e.g.:

```text
-1-_jpg.rf.acf0acc89ce8b077384fb4d7773d0982.jpg
IMG_20230519_130735_jpg.rf.32d4bba2defeb3a4653ce22d7aae6d6c.jpg
```

They do not contain class counts; use `objects` / `class_0..3` in the manifest.

### 4.2 Augmented images

The same suffix convention is used:

```text
_aug{variant}_{category}_{operation}.jpg
```

Examples:

```text
-1-_jpg.rf.acf0acc89ce8b077384fb4d7773d0982_aug1_illum_color_brightness_contrast.jpg
-1-_jpg.rf.acf0acc89ce8b077384fb4d7773d0982_aug2_image_quality_gaussian_blur.jpg
-1-_jpg.rf.acf0acc89ce8b077384fb4d7773d0982_aug3_mild_geometry_translation.jpg
```

The exact sampled parameters are in `aug_parameters` and
`data/dataset_b/augmentation_log.csv`.

## 5. Using the list

1. Join `filename_manifest.csv` with the public source images by `filename`.
2. Verify `image_sha256` if the original file provides a hash.
3. For `subset=augmented`, `source_image` gives the original training frame;
   `aug_variant/category/operation/parameters` describe the transformation.
4. `group_id` is inherited from the original image, so augmented variants never
   cross split boundaries.
5. The counts in the manifest should match the split totals in
   `docs/04_dataset_group_summary.md`.

## 6. Scope note

The manifests are metadata only. Dataset images remain under their original
public licenses and are not redistributed in this repository.