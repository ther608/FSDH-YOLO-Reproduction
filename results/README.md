# Results directory

## Dataset A

| File | Content |
|---|---|
| `dataset_a_per_seed.csv` | 30 keys / 90 rows: per-seed accuracy, Common Eval JSON, checkpoint paths |
| `dataset_a_three_seed_mean_sd.csv` | Mean +/- SD |
| `ap75_per_seed.csv` / `ap75_three_seed.csv` | AP75 |
| `dataset_a_area_ap.csv` | Area-stratified AP (small/non-small) |
| `dataset_a_results.json` | Area AP + confusion matrix + Bootstrap recomputation |
| `dataset_a_confusion_matrices.csv` | Row-normalized confusion matrices |
| `bootstrap_ci_extended.csv` / `bootstrap_results_extended.json` | Bootstrap 95% CI, IoU localization error |
| `shab_comparison_summary.json` / `shab_comparison_seven_modules.md` | Seven-module Common Evaluator summary |
| `common_evaluator/dataset_a/{YOLO11n,FSDH-YOLO_n}/seed{0,1,2}__test_metrics_common.json` | A raw Common Evaluator metrics |
| `latency/dataset_a_*` | A FPS raw/summary CSV |
| `latency/dataset_a_branch_and_pm_latency.csv` | SHEB branch + YOLO11n-PM FPS |

## Dataset B

| File | Content |
|---|---|
| `paper_dataset_b_groupaware_clean_v4_aug_common_v2__main_results.json` | Authoritative source: per-seed, mean+/-SD, paired deltas |
| `...__main_results.md` | Human-readable version |
| `..._task_ap75.csv` | AP75 task |
| `..._task_apsmall_coco.csv` | AP_small(COCO) task |
| `..._task_maturity_ap.csv` | Maturity-class AP task |
| `common_evaluator/dataset_b/{YOLO11n,FSDH-YOLO_n}/seed{0,1,2}__test_metrics_common.json` | B raw Common Evaluator metrics |
| `latency/dataset_b_latency.csv` | B FPS |

## Paper/appendix

| File | Content |
|---|---|
| `paper_3.12_experiment_summary_zh.md` | Original Chinese results ledger (used for the coverage audit; English header added) |
| `Appendix_C/` | Supplementary's 13 per-seed tables (C01-C13) |

## Notes

- `results/` is the source/intermediate layer for paper values; do not replace it
  with native Ultralytics `runs/test/*/metrics_summary.json`.
- A/B AP_small definitions differ; do not compare them numerically.
- Per-seed tables keep 3 decimals; recomputed mean/SD can differ by 0.001 in the
  last digit.