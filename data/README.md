# Data directory

This directory does **not** contain dataset images. It contains, for both public
datasets:

- fixed per-file and per-group split manifests;
- annotation-cleaning and duplicate-box records;
- grouping constraint edges;
- training-only offline augmentation logs/summaries;
- build/audit/verification JSON.

## Sources

| Dataset | Source | Access |
|---|---|---|
| A | Zhang et al., *An RGB-D image dataset for lychee detection and maturity classification for robotic harvesting*, arXiv:2510.16800 | Obtain from the original public source; this repository uses the RGB subset only |
| B | Zhao, *lichi-maturity*, Mendeley Data V1, `doi:10.17632/c3rk9gv4w9.1` | Obtain from Mendeley Data |

## Why images are not uploaded

1. Both datasets are public and have their own licenses/distribution terms.
2. The images are large (A: 2,360 augmented train; B: 4,284 augmented train).
3. Reproducibility is guaranteed by the **fixed split + cleaning/grouping
   manifests + augmentation logs + checkpoints**, without redistributing images.

## Layout

```text
data/
|-- dataset_a/
|   |-- build_summary.json
|   |-- group_manifest.csv
|   |-- split_manifest.csv
|   |-- group_constraint_edges.csv
|   |-- augmentation_log.csv
|   |-- augmentation_scheme.txt
|   `-- README.md
`-- dataset_b/
    |-- build_summary.json
    |-- raw_build_summary.json
    |-- group_manifest.csv
    |-- split_manifest.csv
    |-- image_constraint_edges.csv
    |-- label_cleaning_manifest.csv
    |-- excluded_images.csv
    |-- audit/
    |-- dataset_b_v4_verification.json
    |-- augmentation_summary.json
    |-- augmentation_log.csv
    `-- README.md
```

## Restoration

See `docs/03_dataset_protocol.md` section 3.3:

1. locate each source image by `filename` and verify `image_sha256`;
2. copy/hardlink into train/val/test according to `new_split`;
3. generate training-only augmentation from the per-image log;
4. leave val/test unchanged;
5. verify counts with `group_manifest.csv`.