# Provenance — artifact → source run → hash

This file records where each published artifact came from. The authoritative
machine-readable lists are:

- `weights/MANIFEST.csv` — per-checkpoint params, best epoch, metrics, size, SHA256, source run;
- `weights/SHA256SUMS` — checkpoint hashes;
- `CHECKSUMS.sha256` — every tracked artifact in this repository (except itself);
- `data/dataset_{a,b}/build_summary.json` — dataset construction and split statistics.

## 1. Source code

| Artifact | Source |
|---|---|
| `src/ultralytics/` | Ultralytics 8.4.11 (`https://github.com/ultralytics/ultralytics`) + FSDH-YOLO modifications (SPOConv/SHAB/registration/configs) |
| `train/origin_train.py` | Original paper training entry; only `ROOT`/`sys.path` changed to this repo |
| `train/val.py` | Original independent test wrapper |
| `train/run_reproduction.py` | New A/B convenience runner (does not modify archived scripts) |
| `eval/*` | Original Common Evaluator, latency benchmark, data build/audit, statistics scripts |
| `configs/*` | Copies of `src/ultralytics/cfg/models/11/` FSDH/ablation/config YAMLs |

## 2. Dataset manifests

| Artifact | Source run |
|---|---|
| `data/dataset_a/build_summary.json` | `lychee_groupaware_clean_raw_v2` build |
| `data/dataset_a/group_manifest.csv` | same |
| `data/dataset_a/split_manifest.csv` | same |
| `data/dataset_a/group_constraint_edges.csv` | same |
| `data/dataset_a/augmentation_log.csv` | `lychee_groupaware_clean_aug_v2` augmentation |
| `data/dataset_a/augmentation_scheme.txt` | same |
| `data/dataset_b/build_summary.json` | `lichi_701515_groupaware_clean_v4_raw` build |
| `data/dataset_b/raw_build_summary.json` | `lichi_701515_groupaware_clean_v4_aug` raw summary |
| `data/dataset_b/group_manifest.csv` | raw build |
| `data/dataset_b/split_manifest.csv` | raw build |
| `data/dataset_b/image_constraint_edges.csv` | raw build |
| `data/dataset_b/label_cleaning_manifest.csv` | raw build |
| `data/dataset_b/excluded_images.csv` | raw build |
| `data/dataset_b/audit/*` | raw build protocol/review/box audit |
| `data/dataset_b/dataset_b_v4_verification.json` | independent verification (PASS) |
| `data/dataset_b/augmentation_summary.json` | `..._v4_aug` augmentation |
| `data/dataset_b/augmentation_log.csv` | same |

Build seed: **20260731** for both datasets. Split ratios 70/15/15; group-aware.
No group crosses splits; constraint-edge and exact-hash cross-split counts are 0.

## 3. Annotations

| Artifact | Source |
|---|---|
| `annotations/dataset_a_test_instances.json` | `external_configs/rtdetr_coco_annotations/instances_test.json`; `info.split="test"` added; original SHA256 in `info.original_sha256` |
| `annotations/dataset_b_test_instances.json` | `external_configs/dataset_b_native4_v4/instances_test.json` (already had `info.split`) |

Counts:

- A: 126 images / 2,775 annotations / categories unripe, semi-ripe, ripe;
- B: 212 images / 3,263 annotations / categories green, half, red, young.

## 4. Weights

| Weight | Source run (relative to original workspace) | Best epoch | Common Eval mAP50 / mAP50-95 |
|---|---|---:|---|
| `dataset_a/FSDH-YOLO_n_seed0_best.pt` | `runs/detect/paper_groupaware_clean_v2_reruns/dsA_groupaware_clean_aug_v2__fsd_completion__fsd_shab__seed0` | 63 | 80.931 / 66.670 |
| `dataset_a/FSDH-YOLO_n_seed1_best.pt` | `...__fsd_completion__fsd_shab__seed1` | 74 | 81.500 / 67.140 |
| `dataset_a/FSDH-YOLO_n_seed2_best.pt` | `...__fsd_completion__fsd_shab__seed2` | 52 | 82.866 / 68.080 |
| `dataset_a/YOLO11n_seed0_best.pt` | `...__scale__yolo11n__seed0` | 41 | 80.068 / 65.120 |
| `dataset_a/YOLO11n_seed1_best.pt` | `...__scale__yolo11n__seed1` | 130 | 78.246 / 64.151 |
| `dataset_a/YOLO11n_seed2_best.pt` | `...__scale__yolo11n__seed2` | 44 | 78.637 / 64.016 |
| `dataset_b/FSDH-YOLO_n_seed0_best.pt` | `runs/detect/paper_dataset_b_groupaware_clean_v4_aug_common_v2/dsB_groupaware_clean_v4_aug__comparison__fsdh_yolo_n__seed0` | 82 | 83.993 / 59.683 |
| `dataset_b/FSDH-YOLO_n_seed1_best.pt` | `...__comparison__fsdh_yolo_n__seed1` | 81 | 85.301 / 60.895 |
| `dataset_b/FSDH-YOLO_n_seed2_best.pt` | `...__comparison__fsdh_yolo_n__seed2` | 77 | 84.181 / 60.051 |
| `dataset_b/YOLO11n_seed0_best.pt` | `...__comparison__yolo11n__seed0` | 59 | 85.729 / 61.466 |
| `dataset_b/YOLO11n_seed1_best.pt` | `...__comparison__yolo11n__seed1` | 74 | 84.706 / 61.442 |
| `dataset_b/YOLO11n_seed2_best.pt` | `...__comparison__yolo11n__seed2` | 52 | 84.272 / 60.807 |
| `pretrained/yolo11n.pt` | Ultralytics YOLO11n pretrained | — | — |

All weights are `best.pt` selected by validation mAP50-95. See
`weights/MANIFEST.csv` for SHA256 and exact sizes.

## 5. Results

| Artifact | Source |
|---|---|
| `results/dataset_a_per_seed.csv` | `analysis/paper16_unified_common_protocol/per_seed.csv` |
| `results/dataset_a_three_seed_mean_sd.csv` | `analysis/paper16_unified_common_protocol/three_seed_mean_sd.csv` |
| `results/ap75_*.csv` | `analysis/paper_3_2_ap75_shab/` |
| `results/bootstrap_*` | `analysis/paper_3_4_review/paper35_extended_statistics/` |
| `results/dataset_a_area_ap.csv`, `dataset_a_results.json`, `dataset_a_confusion_matrices.csv` | `analysis/paper27_statistical_supplement/` |
| `results/shab_comparison_*` | `analysis/shab_comparison_common_protocol/` |
| `results/paper_dataset_b_*` | `analysis/paper_dataset_b_groupaware_clean_v4_aug_common_v2__main_results.*` |
| `results/common_evaluator/dataset_{a,b}/**` | `runs/common_eval/.../seed{0,1,2}__test_metrics_common.json` |
| `results/latency/dataset_a_*` | `runs/quality/paper16_{non_fsd,spo_fsd}_uniform_latency/`, `runs/quality/shab_comparison_uniform_latency_fp32_b1_640/` |
| `results/latency/dataset_a_branch_and_pm_latency.csv` | `runs/latency/paper16_shab_branch_subset_and_param_match_v1/`, `runs/latency/paper16_fsdconv_shab_m_confirmation_v1/` |
| `results/latency/dataset_b_latency.csv` | `runs/latency/paper_dataset_b_groupaware_clean_v4_aug_comparison_v1/` |
| `results/Appendix_C/` | `ther608/FSDH-YOLO-Supplementary` (MIT) |
| `results/paper_3.12_experiment_summary_zh.md` | `analysis/paper_3_4_review/paper_3.12_experiment_summary_zh.md` |

## 6. Verification performed

| Check | Result |
|---|---|
| Load all 12 checkpoints with `YOLO(...)` | OK; A nc=3, B nc=4 |
| Build FSDH-YOLO from `configs/yolo11n_FSDConv-SHAB.yaml` | OK; 3,004,777 raw params; forward OK |
| Dataset A FSDH seed0 Common Evaluator | mAP50 80.931%, mAP50-95 66.670%, AP75 76.193%, AP_small 57.577% — matches paper |
| Dataset B FSDH-n seed0 Common Evaluator | mAP50 83.993%, mAP50-95 59.683%, AP75 69.845%, AP_small 32.795% — matches paper |
| `CHECKSUMS.sha256` | all tracked artifacts verified OK |
| Dataset group cross-split violations | A 0, B 0 |
| Dataset constraint-edge cross-split violations | A 0/584, B 0/730 |
| Exact-hash cross-split | A 0/0/0, B 0/0/0 |

## 7. Environment

Windows 11 64-bit; Intel i5-14400F; NVIDIA RTX 5060 Ti 16 GB; Python 3.10;
CUDA 12.8; PyTorch 2.10+cu128; Ultralytics 8.4.11 (this repo's `src/`).
