# 02. Existing GitHub repositories and their structure

Search date: 2026-10-17
Account: `ther608` (`gh auth status` authenticated)

## 0. Summary

Two GitHub repositories are directly related to the FSDH-YOLO paper:

| Repository | Visibility | Purpose | Relation to this repo |
|---|---|---|---|
| [`ther608/FSDH-YOLO-Supplementary`](https://github.com/ther608/FSDH-YOLO-Supplementary) | Public | Supplementary materials: Appendix C per-seed results, evaluation protocol, Android deployment, ONNX | This repo does **not** duplicate it; it adds training code, PT checkpoints and split manifests |
| `ther608/fsdh-yolo-deploy` | Private | Android/TFLite/ONNX deployment project and export records | Same deployment lineage as Supplementary's Android part; not copied here |

A public GitHub search found no other FSDH-YOLO training-code repository. One
unrelated litchi-maturity YOLOv8 training repository was found
(`QQ767172261/Deep-learning-target-detection-algorithms-use-litchi-maturity-datasets-to-train-YOLOv8-models-and-ev`,
0 stars, no author/code relation).

---

## 1. `FSDH-YOLO-Supplementary` (public)

GitHub metadata:

| Item | Value |
|---|---|
| Repository | `ther608/FSDH-YOLO-Supplementary` |
| Default branch | `main` |
| Created | 2026-10-11 08:32:52 UTC |
| Updated | 2026-10-11 08:59:45 UTC |
| GitHub size | 19,148 KB (~18.7 MB) |
| License | MIT (root `LICENSE`; separate `LICENSE-docs`) |
| Tree entries | 168 |
| Forks | 0 |

### 1.1 Top-level structure

```text
FSDH-YOLO-Supplementary/
|-- README.md                       # scope, provided/not provided, headline results
|-- CITATION.cff
|-- LICENSE                         # MIT
|-- LICENSE-docs
|-- .gitignore
|-- .gitattributes
|-- Appendix_C/                     # 17 files
|   |-- Appendix_C.md               # 13 per-seed tables (C01-C13)
|   |-- README.md
|   `-- per_seed/
|       |-- C01_multiscale_feature_enhancement_per_seed.csv
|       |-- C02_ablation_per_seed.csv
|       |-- C03_small_object_AP_per_seed.csv
|       |-- C04_AP75_localization_per_seed.csv
|       |-- C05_bbox_localization_error_per_seed.csv
|       |-- C06_maturity_class_AP_per_seed.csv
|       |-- C07_confusion_matrix_row_normalized.csv
|       |-- C08_general_detector_comparison_per_seed.csv
|       |-- C09_recent_method_comparison_per_seed.csv
|       |-- C10_cross_dataset_per_seed.csv
|       |-- C11_yolo11_scale_selection_per_seed.csv
|       |-- C12_spdconv_vs_fsdconv_per_seed.csv
|       `-- C13_sheb_branch_ablation_per_seed.csv
|-- android_deployment/             # 141 entries
|   |-- README.md
|   |-- PROVENANCE.md               # artifact hashes and source-run mapping
|   |-- android_app/                # Kotlin Android Studio project
|   |   `-- app/src/main/assets/
|   |       |-- fsdh_yolo_n_rerun_seed0_640_fp32.onnx   (12.28 MB)
|   |       `-- yolo11n_baseline_640_fp32.onnx          (10.61 MB)
|   |-- deployment_evaluation_results/
|   |   |-- README.md
|   |   |-- continuous_camera_sessions.csv
|   |   |-- fixed_input_latency.csv
|   |   |-- deployment_footprint.md
|   |   `-- raw_records/bench/*.json
|   |-- extended_paper_materials/   # section 3.9 / A.7 / B.5 text
|   `-- scripts/                    # export and verification scripts
|-- evaluation_protocol/
|   `-- README.md                   # metrics, area thresholds, confusion matrix, Bootstrap, FPS
`-- reproduction_notes/
    `-- README.md                   # environment, data, statistics, provenance, known issues
```

### 1.2 What it provides / does not provide

The README states:

| Provided | Not provided |
|---|---|
| Appendix C: per-seed results for all 13 experiment tables | Full training/evaluation framework source |
| Android package: Kotlin app, protocol, raw measurements, ONNX | Datasets |
| Evaluation protocol and reproduction notes | PyTorch `.pt` checkpoints |

This is exactly the gap this repository fills: **source + training/evaluation
scripts + PT checkpoints + split manifests**.

### 1.3 Key documents

- `evaluation_protocol/README.md`:
  - 640x640 square inference; conf floor 0.001; class-aware NMS IoU 0.70;
    max_det 300;
  - COCO 101-point AP; P/R at the class-mean-F1-max confidence;
  - `AP_small(640)` = original-image `[0,4096) px2`; `COCO AP_s` = `<1024 px2`;
  - confusion matrix conf 0.25 / IoU 0.50, row-normalized, Miss/FN column;
  - Bootstrap 10,000 image-wise paired draws; FPS protocol (FP32, b1, 640,
    warmup 100, timing 500, forward+NMS).
- `reproduction_notes/README.md`:
  - environment: Windows 11, i5-14400F, RTX 5060 Ti, Python 3.10, CUDA 12.8,
    PyTorch 2.10+cu128, Ultralytics 8.4.11;
  - dataset image counts, classes, splits and augmentation summary;
  - code snippets for recomputing statistics;
  - provenance map;
  - known issues: Appendix B.2 label/last digit, two paired differences in
    section 3.8, two deployment-appendix cells;
  - confirms the two shifted 3.13 per-seed tables were fixed.

> Some notes target 3.13. Current 3.23 has fixed most of them. See
> `docs/01_coverage_audit.md` for the 3.23 re-check.

---

## 2. `fsdh-yolo-deploy` (private)

GitHub metadata:

| Item | Value |
|---|---|
| Repository | `ther608/fsdh-yolo-deploy` |
| Visibility | Private |
| Default branch | `master` |
| Created | 2026-10-06 08:41:24 UTC |
| GitHub size | 88,434 KB (~86.4 MB) |

Structure:

```text
fsdh-yolo-deploy/
|-- README.md                       # 34.8 KB, deployment notes
|-- .github/workflows/coreml-placement.yml
|-- android/app/
|   |-- build.gradle.kts
|   `-- src/main/
|       |-- assets/
|       |   |-- fsdh_yolo_n_rerun_seed0_640_fp32.onnx   (12.28 MB)
|       |   |-- fsdh_yolo_n_rerun_seed0_640_fp32.tflite (12.19 MB)
|       |   |-- yolo11n_baseline_640_fp32.onnx          (10.61 MB)
|       |   `-- yolo11n_baseline_640_fp32.tflite         (10.52 MB)
|       |-- java/com/lychee/fsdh/
|       |   |-- MainActivity.kt / BenchActivity.kt / OverlayView.kt
|       |   |-- bench/ (Benchmark.kt, LiveSession.kt, PowerMonitor.kt, Stats.kt)
|       |   |-- camera/YuvToRgb.kt
|       |   `-- infer/ (Detector.kt, OrtEngine.kt, TfliteEngine.kt, Preprocessor.kt, ...)
|       `-- res/
|-- exported/
|   |-- fsdh_yolo_n_rerun_seed0_640_fp32.tflite
|   |-- fsdh_yolo_n_rerun_seed0_640_fp32_opset17.onnx
|   |-- fsdh_yolo_n_rerun_seed0_640_fp32_saved_model/
|   |-- yolo11n_baseline_640_fp32.tflite
|   `-- yolo11n_baseline_640_fp32_saved_model/
|-- records/                        # decode/export/verification JSON and logs
`-- scripts/build_apk.ps1
```

It is a deployment/export repository, not a training repository: no dataset
splits, training code or per-seed accuracy tables. Its ONNX/TFLite models are
the `fsdh_yolo_n_rerun_seed0` lineage, same as the Supplementary Android package.
Because it is private, the public paper should link to
`FSDH-YOLO-Supplementary/android_deployment` instead.

---

## 3. Division of labour

| Repository | Audience | Contains | Does not contain |
|---|---|---|---|
| `FSDH-YOLO-Supplementary` (existing) | Reviewers checking numbers | Appendix C, evaluation protocol, Android package, ONNX, raw device records | Training code, PT checkpoints, datasets |
| `FSDH-YOLO-Reproduction` (this) | Reproducing training/evaluation | Modified source, configs, train/eval scripts, fixed split manifests, PT checkpoints, result index | Dataset images, Android project |
| `fsdh-yolo-deploy` (private) | Internal deployment | APK build, TFLite/CoreML experiments, raw records | Public supplementary material (use the public Supplementary) |

Suggested Data/Code Availability text:

- Supplementary (Appendix C + Android + protocol):
  `https://github.com/ther608/FSDH-YOLO-Supplementary`
- Reproduction (code + configs + scripts + weights + split manifests):
  `https://github.com/ther608/FSDH-YOLO-Reproduction`
- Datasets: public source DOIs; not redistributed.