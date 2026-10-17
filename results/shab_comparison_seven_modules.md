# SHEB vs six multi-scale modules under a unified protocol

## Evaluation protocol

- Compared modules: SHEB, ASPP, SPP, DilatedConv, RepConv, MultiScaleDWConv and RFB (7 structures).
- Accuracy: all metrics come from the unified Common Evaluator on the Dataset A test split.
- Inference: `imgsz=640`, confidence floor `0.001`, NMS IoU `0.7`, `max_det=300`.
- P/R: at the confidence maximizing class-mean F1, with class-aware matching IoU >= 0.5.
- Mean/error: arithmetic mean +/- sample SD (`ddof=1`) over 3 seeds.
- Params: Ultralytics complexity convention, consistent with the paper SHEB `2.995793 M`.
- FPS: not a Common Evaluator accuracy metric; from a separate uniform batch=1, FP32, 640x640, fused-forward + class-aware NMS benchmark.
- In this same-machine FPS round SHEB is `100.54+/-0.99`, not to be mixed with the historical `122.01+/-1.17`.

## Per-seed results

| Module | Seed | P/% | R/% | mAP50/% | mAP50-95/% | Params/M | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| SHEB | 0 | 78.173 | 79.970 | 83.031 | 67.388 | 2.996 | 101.68 |
| SHEB | 1 | 78.297 | 79.081 | 79.013 | 65.106 | 2.996 | 100.01 |
| SHEB | 2 | 78.556 | 77.513 | 81.233 | 66.152 | 2.996 | 99.91 |
| ASPP | 0 | 77.103 | 79.754 | 80.793 | 65.918 | 3.419 | 97.91 |
| ASPP | 1 | 75.772 | 76.710 | 78.881 | 64.349 | 3.419 | 98.97 |
| ASPP | 2 | 79.004 | 78.772 | 81.939 | 66.235 | 3.419 | 97.15 |
| SPP | 0 | 78.360 | 79.066 | 80.615 | 66.164 | 2.829 | 108.81 |
| SPP | 1 | 77.198 | 79.333 | 80.508 | 65.587 | 2.829 | 111.65 |
| SPP | 2 | 77.144 | 81.029 | 82.641 | 67.505 | 2.829 | 110.97 |
| DilatedConv | 0 | 76.499 | 80.966 | 81.543 | 66.357 | 2.688 | 111.19 |
| DilatedConv | 1 | 76.432 | 79.850 | 80.927 | 65.889 | 2.688 | 109.19 |
| DilatedConv | 2 | 76.146 | 81.447 | 80.995 | 66.397 | 2.688 | 109.38 |
| RepConv | 0 | 77.828 | 79.842 | 81.065 | 65.703 | 3.468 | 110.87 |
| RepConv | 1 | 75.614 | 80.702 | 80.478 | 65.892 | 3.468 | 115.69 |
| RepConv | 2 | 77.658 | 77.821 | 79.662 | 65.344 | 3.468 | 115.53 |
| MultiScaleDWConv | 0 | 78.327 | 79.506 | 80.934 | 66.452 | 2.742 | 101.78 |
| MultiScaleDWConv | 1 | 77.500 | 80.340 | 81.745 | 66.628 | 2.742 | 103.35 |
| MultiScaleDWConv | 2 | 76.525 | 79.555 | 80.389 | 65.992 | 2.742 | 103.85 |
| RFB | 0 | 77.499 | 81.230 | 82.507 | 67.348 | 2.947 | 100.37 |
| RFB | 1 | 76.363 | 79.522 | 79.093 | 65.174 | 2.947 | 92.88 |
| RFB | 2 | 77.355 | 77.728 | 77.711 | 64.372 | 2.947 | 82.79 |

## Three-seed mean results

| Module | P/% | R/% | mAP50/% | mAP50-95/% | Params/M | FPS |
|---|---:|---:|---:|---:|---:|---:|
| SHEB | 78.342 ± 0.196 | 78.854 ± 1.244 | 81.092 ± 2.012 | 66.215 ± 1.142 | 2.996 | 100.54 ± 0.99 |
| ASPP | 77.293 ± 1.624 | 78.412 ± 1.554 | 80.538 ± 1.545 | 65.501 ± 1.010 | 3.419 | 98.01 ± 0.92 |
| SPP | 77.567 ± 0.687 | 79.809 ± 1.065 | 81.255 ± 1.202 | 66.419 ± 0.984 | 2.829 | 110.47 ± 1.48 |
| DilatedConv | 76.359 ± 0.187 | 80.754 ± 0.819 | 81.155 ± 0.338 | 66.214 ± 0.283 | 2.688 | 109.92 ± 1.10 |
| RepConv | 77.033 ± 1.232 | 79.455 ± 1.479 | 80.402 ± 0.704 | 65.646 ± 0.278 | 3.468 | 114.03 ± 2.74 |
| MultiScaleDWConv | 77.451 ± 0.902 | 79.800 ± 0.468 | 81.022 ± 0.682 | 66.357 ± 0.328 | 2.742 | 102.99 ± 1.08 |
| RFB | 77.072 ± 0.618 | 79.493 ± 1.751 | 79.770 ± 2.469 | 65.631 ± 1.540 | 2.947 | 92.01 ± 8.82 |

## Brief comparison

- SHEB has the highest mean Precision, 78.342%.
- DilatedConv has the highest mean Recall, 80.754%.
- SPP has the highest mean mAP50 and mAP50-95, 81.255% and 66.419%, only 0.163 and 0.204 pp above SHEB.
- These AP differences are within three-seed variation, supporting comparable performance, not a claim that SPP is significantly better than SHEB.

## Source locations

- Common Evaluator per-seed JSON: `runs/common_eval/paper16_unified_protocol/shab*`。
- Uniform latency per-seed results: `runs/quality/shab_comparison_uniform_latency/uniform_latency_results.csv`。
- Machine-readable summary: `analysis/shab_comparison_common_protocol/summary.json`。
