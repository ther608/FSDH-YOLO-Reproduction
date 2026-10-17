# 09. Paper table <-> result file <-> checkpoint

## 1. Dataset A

| Paper table | Content | Result files | Checkpoints (this repo) |
|---|---|---|---|
| Table 3 | multi-scale feature enhancement | `results/shab_comparison_summary.json`, `results/common_evaluator/dataset_a/*`, `results/latency/dataset_a_*` | FSDH/YOLO11n; other module weights not bundled |
| Table 4 | ablation | `results/dataset_a_per_seed.csv`, `dataset_a_three_seed_mean_sd.csv` | A FSDH x3, YOLO11n x3; single-module weights not bundled |
| Table 5 | small-object area stratification | `results/dataset_a_area_ap.csv`, `dataset_a_results.json`, `bootstrap_ci_extended.csv` | A FSDH x3, YOLO11n x3 |
| Table 6 | AP75 | `results/ap75_per_seed.csv`, `ap75_three_seed.csv` | A FSDH x3, YOLO11n x3 |
| Table 7 | common-detection IoU error | `results/bootstrap_ci_extended.csv`, `bootstrap_results_extended.json` | A FSDH x3, YOLO11n x3 |
| Table 8 | maturity-class AP | `results/common_evaluator/dataset_a/*`, `bootstrap_ci_extended.csv` | A FSDH x3, YOLO11n x3 |
| Table 10 | general detectors | `results/dataset_a_per_seed.csv` (YOLO11n/FSDH rows), `results/latency/` | only YOLO11n/FSDH; external weights in original workspace |
| Table 11 | recent fruit models | `results/dataset_a_per_seed.csv`, `results/latency/dataset_a_branch_and_pm_latency.csv` | only FSDH; reproduced weights in original workspace |
| Appendix B.1 | YOLO11 n/s/m scale | `results/dataset_a_per_seed.csv`, `results/latency/dataset_a_non_fsd_*` | only n |
| Appendix B.2 | SPDConv vs FSDConv | `results/Appendix_C/C12_*`, `results/latency/dataset_a_shab_comparison_*` | only FSDH; SPD weights in original workspace |
| Appendix B.3 | SHEB branches | `results/Appendix_C/C13_*`, `results/latency/dataset_a_branch_and_pm_latency.csv` | only full FSDH; branch weights in original workspace |
| Appendix B.4 | Bootstrap | `results/bootstrap_ci_extended.csv`, `bootstrap_results_extended.json` | A checkpoints |
| Appendix C | per-seed tables | `results/Appendix_C/` (C01-C13) | corresponding runs |

## 2. Dataset B

| Paper table | Content | Result files | Checkpoints |
|---|---|---|---|
| Table 12 | overall P/R/mAP50/mAP50-95/AP75/AP_small | `results/paper_dataset_b_...__main_results.json/.md`, `common_evaluator/dataset_b/*` | B FSDH-n x3, YOLO11n x3 |
| Table 13 | per-class AP50/AP50-95 | same + `..._task_maturity_ap.csv` | B FSDH-n x3, YOLO11n x3 |
| Appendix C.10 | cross-dataset per-seed | `results/Appendix_C/C10_*` | B checkpoints |
| FPS (not a paper table column) | YOLO11n 165.56+/-3.10 / FSDH 140.06+/-2.34 | `results/latency/dataset_b_latency.csv` | B checkpoints |

## 3. Deployment (not in this repository)

| Paper table | Content | Location |
|---|---|---|
| Table 14 | Android fixed-input/continuous-camera | Supplementary `android_deployment/` |
| Appendix A.7 | deployment protocol | same |
| Appendix B.5 | three 60 s runs + diagnostic rerun | same |

## 4. Checkpoint <-> result

| Checkpoint | Results |
|---|---|
| `weights/dataset_a/FSDH-YOLO_n_seed{0,1,2}_best.pt` | all A FSDH rows (Tables 3-8, 10-11, B.4) |
| `weights/dataset_a/YOLO11n_seed{0,1,2}_best.pt` | all A YOLO11n rows |
| `weights/dataset_b/FSDH-YOLO_n_seed{0,1,2}_best.pt` | B FSDH rows (Tables 12/13, C10) |
| `weights/dataset_b/YOLO11n_seed{0,1,2}_best.pt` | B YOLO11n rows |
| `weights/pretrained/yolo11n.pt` | initialization for all training |

## 5. Re-evaluation entry

```bash
# single checkpoint
PYTHONPATH=src python eval/eval_ultralytics_common_evaluator.py ...

# one-command training + evaluation
python train/run_reproduction.py --dataset a --dataset-root ...
```

Exact seed-0 reproduction is documented in `docs/05_weight_inventory.md` section 4.