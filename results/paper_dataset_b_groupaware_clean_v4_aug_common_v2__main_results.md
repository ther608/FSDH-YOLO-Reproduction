# Dataset B v4 aug: YOLO11n vs FSDH-YOLO-n Common Evaluator comparison

Generated: `2026-09-21T21:18:05+08:00`  
Compared models: YOLO11n and FSDH-YOLO-n (FSDConv+SHAB); seeds 0/1/2.
Statistics: mean +/- sample SD (ddof=1); all accuracy metrics recomputed by the same Common Evaluator.

## 1. Unified evaluation protocol

- split: `test`; input size: `640x640`.
- prediction confidence floor：`0.001`；class-aware NMS IoU：`0.7`；max_det：`300`。
- AP-small: `AP_small_COCO` uses the standard COCO small definition (original-image area < 1024 pixels); all images are 640x640, so Dataset A's area x4 convention is not used.
- Frozen test GT: 212 images, 3,263 instances; COCO annotation SHA-256: `667b9746d3478be36a324e8093a0abfb03b09c99c7b12f461ad8c20e50dbbb4f`.
- The four stages follow `young -> green -> half -> red`; the two intermediate classes `green` and `half` are reported separately (not merged/averaged), and Dataset A's `semi-ripe AP` naming is not used.
- P/R: computed at the confidence maximizing mean F1 over four classes; AP uses the full low-threshold prediction set.

## 2. Main paper-table results

| Model | P/% | R/% | mAP50/% | mAP50-95/% | AP75/% | AP-small COCO/% |
|---|---|---|---|---|---|---|
| YOLO11n | 81.097±1.471 | 82.504±2.222 | 84.902±0.748 | 61.238±0.374 | 70.577±0.443 | 32.332±1.812 |
| FSDH-YOLO-n | 81.073±2.128 | 82.820±0.674 | 84.492±0.707 | 60.210±0.621 | 70.294±0.630 | 32.524±1.329 |

## 3. Task-specific results

### 3.1 Strict localization: AP75

| Model | AP75 mean +/- sample SD/% | seed0/seed1/seed2/% |
|---|---|---|
| YOLO11n | 70.577±0.443 | 70.069 / 70.882 / 70.779 |
| FSDH-YOLO-n | 70.294±0.630 | 69.845 / 71.015 / 70.023 |

### 3.2 Small objects: COCO AP-small

| Model | AP-small COCO mean +/- sample SD/% | seed0/seed1/seed2/% |
|---|---|---|
| YOLO11n | 32.332±1.812 | 30.335 / 32.790 / 33.870 |
| FSDH-YOLO-n | 32.524±1.329 | 32.795 / 33.697 / 31.080 |

- Note: images are 640x640, so the original COCO small-area threshold `<32^2=1024 px^2` is used; Dataset A's `AP_small_area4` is not used.

### 3.3 Maturity classification: per-class AP50 and AP50-95

| Model | Maturity | AP50 mean +/- SD/% | AP50-95 mean +/- SD/% | per-seed AP50/% | per-seed AP50-95/% |
|---|---|---|---|---|---|
| YOLO11n | green | 89.155±0.334 | 64.945±0.498 | 89.267 / 88.780 / 89.419 | 64.526 / 64.815 / 65.495 |
| YOLO11n | half | 70.062±2.834 | 57.935±2.923 | 73.180 / 69.367 / 67.641 | 61.218 / 56.976 / 55.612 |
| YOLO11n | red | 87.199±0.344 | 57.507±1.198 | 87.401 / 86.802 / 87.395 | 56.132 / 58.060 / 58.328 |
| YOLO11n | young | 93.192±0.631 | 64.565±1.175 | 93.067 / 93.876 / 92.632 | 63.987 / 65.916 / 63.791 |
| FSDH-YOLO-n | green | 90.947±0.674 | 66.475±0.336 | 90.268 / 91.615 / 90.959 | 66.244 / 66.861 / 66.320 |
| FSDH-YOLO-n | half | 66.015±1.698 | 53.660±0.842 | 64.912 / 67.970 / 65.163 | 52.726 / 54.360 / 53.894 |
| FSDH-YOLO-n | red | 86.815±0.352 | 55.831±0.861 | 86.717 / 87.206 / 86.522 | 55.308 / 56.825 / 55.359 |
| FSDH-YOLO-n | young | 94.190±0.192 | 64.873±0.580 | 94.077 / 94.413 / 94.082 | 64.453 / 65.534 / 64.632 |

## 4. Per-seed overall results

| Model | seed | P/% | R/% | mAP50/% | mAP50-95/% | AP75/% | AP-small COCO/% |
|---|---|---|---|---|---|---|---|
| YOLO11n | 0 | 80.060 | 84.385 | 85.729 | 61.466 | 70.069 | 30.335 |
| YOLO11n | 1 | 82.781 | 80.053 | 84.706 | 61.442 | 70.882 | 32.790 |
| YOLO11n | 2 | 80.451 | 83.075 | 84.272 | 60.807 | 70.779 | 33.870 |
| FSDH-YOLO-n | 0 | 79.807 | 83.488 | 83.993 | 59.683 | 69.845 | 32.795 |
| FSDH-YOLO-n | 1 | 83.530 | 82.141 | 85.301 | 60.895 | 71.015 | 33.697 |
| FSDH-YOLO-n | 2 | 79.882 | 82.831 | 84.181 | 60.051 | 70.023 | 31.080 |

## 5. FSDH-YOLO minus YOLO11n paired changes

| Metric | Paired mean +/- sample SD/pp | seed0/seed1/seed2/pp |
|---|---|---|
| P | -0.025±0.688 | -0.254 / +0.748 / -0.568 |
| R | +0.316±1.570 | -0.897 / +2.089 / -0.244 |
| mAP50 | -0.410±1.198 | -1.735 / +0.595 / -0.090 |
| mAP50-95 | -1.028±0.662 | -1.783 / -0.547 / -0.755 |
| AP75 | -0.282±0.447 | -0.224 / +0.133 / -0.756 |
| AP-small (COCO) | +0.192±2.697 | +2.460 / +0.907 / -2.790 |
| green AP50 | +1.792±0.943 | +1.001 / +2.836 / +1.541 |
| green AP | +1.529±0.632 | +1.718 / +2.045 / +0.825 |
| half AP50 | -4.047±3.695 | -8.268 / -1.396 / -2.478 |
| half AP | -4.275±3.679 | -8.492 / -2.615 / -1.719 |
| red AP50 | -0.385±0.689 | -0.684 / +0.404 / -0.874 |
| red AP | -1.676±1.138 | -0.825 / -1.235 / -2.968 |
| young AP50 | +0.999±0.456 | +1.010 / +0.537 / +1.450 |
| young AP | +0.308±0.627 | +0.466 / -0.382 / +0.841 |

## 6. Extended quality metrics

| Model | max_F1 | F1_threshold | AR300 | corr_score_iou_tp | ECE_IoU50 | predictions | TP_matches |
|---|---|---|---|---|---|---|---|
| YOLO11n | 81.767±0.387 | 0.443±0.067 | 72.155±0.442 | 0.534±0.007 | 0.023±0.003 | 14767.333±1219.936 | 3180.667±5.033 |
| FSDH-YOLO-n | 81.922±0.798 | 0.435±0.073 | 70.036±0.661 | 0.565±0.011 | 0.022±0.002 | 16426.000±707.010 | 3167.000±8.660 |

## 7. Complexity and uniform latency (not Common accuracy metrics)

| Model | Params/M | GFLOPs | Total latency/ms | FPS |
|---|---|---|---|---|
| YOLO11n | 2.582932±0.000000 | N/A | N/A | N/A |
| FSDH-YOLO-n | 2.997284±0.000000 | N/A | N/A | N/A |

## 8. Traceability

- Common Evaluator root: `<original-workspace>\runs\common_eval\paper_dataset_b_groupaware_clean_v4_aug_common_v2`
- Training/initialization root: `<original-workspace>\runs\detect\paper_dataset_b_groupaware_clean_v4_aug_common_v2`
- Independent test root: `<original-workspace>\runs\test\paper_dataset_b_groupaware_clean_v4_aug_common_v2`
- Uniform latency root: not provided
- In this Dataset B two-model comparison, FSDH uses only Ultralytics native same-name/same-shape transfer, with no semantic layer-index remapping.
- The pre-model seed and native-transfer coverage for each seed are recorded in `initialization_summary.json`.
- Every metric can be traced to the corresponding `seed{k}__test_metrics_common.json` and `seed{k}__test_predictions_coco.json`.
- This file reports numbers and paired directions only; mechanism/statistical-significance conclusions should be written after manual review.
