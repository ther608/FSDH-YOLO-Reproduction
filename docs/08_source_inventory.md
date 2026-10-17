# 08. FSDH-YOLO source, configuration and script inventory

## 1. Model code

### 1.1 FSDConv = `SPOConv`

| Item | Value |
|---|---|
| File | `src/ultralytics/nn/modules/advanced.py` |
| Class | `SPOConv` (line 176) |
| Paper name | FSDConv (Feature-preserving Space-to-Depth Convolution) |
| Implementation | `nn.PixelUnshuffle(2)` -> `Conv(c1*4, c2, k=3, s=1)` (Conv-BN-SiLU) |
| Input/output | `B x c1 x H x W` (H/W even) -> `B x c2 x H/2 x W/2` |
| Parameters | Same 2.584 M class as the standard stride-2 stem; negligible overhead |
| Compatibility | Class name remains `SPOConv` because the paper checkpoints store that name; do not rename it |

```python
class SPOConv(nn.Module):
    def __init__(self, c1, c2, kernel_size=3):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.cv = Conv(c1 * 4, c2, kernel_size, 1)

    def forward(self, x):
        return self.cv(self.unshuffle(x))
```

The convolution stride is 1; spatial downsampling is done by PixelUnshuffle and
the convolution only projects channels.

### 1.2 SHEB = `SHAB`

| Item | Value |
|---|---|
| File | `src/ultralytics/nn/modules/small_object_modules.py` |
| Class | `SHAB` (line 46) |
| Paper name | SHEB (Spatial Hierarchy Enhancement Block) |
| Branches | `L`: local 3x3 depthwise; `M`: 5x5 dilated depthwise (dilation=2, theoretical RF 9x9); `G`: GAP -> 1x1 -> ReLU -> 1x1 -> Sigmoid channel gate |
| Fusion | concat branches -> 1x1 Conv -> ReLU; add input residual |
| Channels | `mid_c = max(c1 // 4, 16)` |
| Insertion | after backbone stages P3/P4/P5, before the Neck |
| Compatibility | `branches` accepts legacy int (1=L, 2=LM, 3=LMG) and new selectors (L/M/G/LM/...) |

```python
class SHAB(nn.Module):
    def __init__(self, c1, branches=3, *_):
        ...
        self.local = ...   # 1x1 -> ReLU -> 3x3 DWConv
        self.mid   = ...   # 1x1 -> ReLU -> 5x5 DWConv(dilation=2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc   = ...  # 1x1 -> ReLU -> 1x1 -> Sigmoid
        self.fuse = ...    # 1x1 Conv(c1*branches -> c1) + ReLU

    def forward(self, x):
        outputs = []
        if "L" in branch_names: outputs.append(self.local(x))
        if "M" in branch_names: outputs.append(self.mid(x))
        if "G" in branch_names: outputs.append(self.global_fc(self.global_pool(x)) * x)
        return x + self.fuse(torch.cat(outputs, dim=1))
```

### 1.3 Registration and parsing

| File | Role | Key lines |
|---|---|---|
| `src/ultralytics/nn/modules/__init__.py` | exports `SPOConv`, `SHAB` and controls | 123, 142, 227 |
| `src/ultralytics/nn/tasks.py` | `parse_model` channel rules for `SPOConv`/`SPDConv` and channel-invariant `SHAB` | 61, 102, 1906, 1965 |
| `src/ultralytics/nn/modules/advanced.py` | `SPOConv`, `SPDConv`, etc. | 176 |
| `src/ultralytics/nn/modules/small_object_modules.py` | `SHAB`, ASPP, DilatedConv, RepConvBlock, MultiScaleDWConv, ReceptiveFieldBlock, CFSB | 46, 138+ |

```python
elif m in frozenset({SPDConv, SPOConv}):
    c2 = make_divisible(min(args[0], max_channels) * width, 8)
    c1 = ch[f]
    args = [c1, c2, *args[1:]]
```

```python
elif m in frozenset({SHAB, ASPP, CFSB, DilatedConv,
                     MultiScaleDWConv, ReceptiveFieldBlock, RepConvBlock}):
    c2 = ch[f]
    args = [ch[f], *args]
```

## 2. Model YAMLs (`configs/`)

### 2.1 Main model

`configs/yolo11n_FSDConv-SHAB.yaml`:

```yaml
nc: 3
scale: n
scales:
  n: [0.50, 0.25, 1024]
backbone:
  - [-1, 1, SPOConv, [64, 3]]   # FSDConv@Input
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, SHAB, []]           # SHEB after P3
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, SHAB, []]           # SHEB after P4
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]
  - [-1, 1, SHAB, []]           # SHEB after P5
head: ...                       # standard YOLO11 FPN-PAN + Detect
```

`nc: 3` is the Dataset A default; Ultralytics overrides the head class count from
the data YAML, so Dataset B (nc=4) reuses the same file.

### 2.2 Ablation/control configs

| File | Experiment | Notes |
|---|---|---|
| `yolo11n_FSDConv-SHAB.yaml` | main | FSDConv + full SHEB |
| `yolo11n_FSDConv.yaml` | Table 4 | FSDConv only |
| `yolo11n_SHAB.yaml` | Table 4 / B.3 | full SHEB only (L+M+G) |
| `yolo11n_SHAB-local.yaml` | B.3 | SHEB-L |
| `yolo11n_SHAB-mid.yaml` | B.3 | SHEB-M |
| `yolo11n_SHAB-global.yaml` | B.3 | SHEB-G |
| `yolo11n_SHAB-local-mid.yaml` | B.3 | SHEB-LM |
| `yolo11n_SHAB-local-global.yaml` | B.3 | SHEB-LG |
| `yolo11n_SHAB-mid-global.yaml` | B.3 | SHEB-MG |
| `yolo11n_FSDConv-SHAB-mid.yaml` | B.3 | FSDConv + SHEB-M |
| `yolo11n_SPDConv.yaml` | B.2 | SPDConv only |
| `yolo11n_SPDConv-SHAB.yaml` | B.2 | SPDConv + full SHEB |
| `yolo11n_SPDConv-SHAB-mid.yaml` | reference | SPDConv + SHEB-M |
| `yolo11n_PM.yaml` | Table 4 | capacity-matched YOLO11n-PM (width 0.25->0.27, max 1024->992, depth 0.50) |
| `yolo11n_capacity_matched.yaml` | reference | alternative capacity control |
| `yolo11n_ASPP.yaml` | Table 3 | ASPP control |
| `yolo11n_SPP-Compare.yaml` | Table 3 | SPP control |
| `yolo11n_DilatedConv.yaml` | Table 3 | single dilated conv control |
| `yolo11n_RepConv.yaml` | Table 3 | RepConv control |
| `yolo11n_MultiScaleDWConv.yaml` | Table 3 | 3/5/7 DWConv control |
| `yolo11n_RFB.yaml` | Table 3 | RFB control |
| `yolo11.yaml` | baseline | YOLO11 architecture |

### 2.3 Run-name mapping

| Paper model | Config | Dataset A run base | Dataset B run base |
|---|---|---|---|
| YOLO11n | `yolo11n.pt` | `dsA_groupaware_clean_aug_v2__scale__yolo11n` | `dsB_groupaware_clean_v4_aug__comparison__yolo11n` |
| FSDH-YOLO | `yolo11n_FSDConv-SHAB.yaml` | `dsA_groupaware_clean_aug_v2__fsd_completion__fsd_shab` | `dsB_groupaware_clean_v4_aug__comparison__fsdh_yolo_n` |
| FSDConv only | `yolo11n_FSDConv.yaml` | `...__fsd_completion__fsd_only` | - |
| SHEB only | `yolo11n_SHAB.yaml` | `...__ablation__shab` | - |
| YOLO11n-PM | `yolo11n_PM.yaml` | `...__capacity_control__yolo11n_pm` | - |
| SPDConv+SHEB | `yolo11n_SPDConv-SHAB.yaml` | `...__spdcontrol__spdconv_shab` | - |
| SHEB-L/M/G/LM/LG/MG | `yolo11n_SHAB-*.yaml` | `...__shabbranch__<branch>` | - |

Full run names are in `results/dataset_a_per_seed.csv` (`run_base`) and
`weights/MANIFEST.csv` (`source_run`).

## 3. Training scripts

| Script | Purpose | Key points |
|---|---|---|
| `train/origin_train.py` | paper training entry | argparse; `init_seeds(seed+1, deterministic=True)`; `YOLO(model)`; optional `--weights`/`--weights-map`; `model.train(...)`; writes `final_summary.txt`, `initialization_summary.json`, `results.csv`, `weights/best.pt` |
| `train/val.py` | native Ultralytics val/test | reads `best.pt`; writes `metrics_summary.json`; for checkpoint traceability, not paper table values |
| `train/save_yolo_final_summary.py` | final summary | environment/model/validation summary |
| `train/run_reproduction.py` | new A/B entry | runtime YAML; train -> val -> Common Eval; `--plan-only`, `--skip-train`, `--latency` |
| `train/run_all_models.ps1` | archived Dataset B campaign | 300 epochs, batch 12, patience 100, seeds 0/1/2; provenance checks; absolute paths |
| `train/run_all_train_val.ps1` | archived Dataset B legacy campaign | same, plus quality/Common Eval/latency |

`origin_train.py`:

1. calls `init_seeds(seed+1, deterministic=True)` before constructing `YOLO`;
2. if `--weights` is given, `--weights-map none` uses native name/shape transfer;
   other maps use semantic layer remapping;
3. `model.train(...)` fixes `imgsz=640`, `close_mosaic=0`, `device="0"`,
   `optimizer="SGD"`, `amp=True`, `deterministic=True`, `exist_ok=False`;
4. writes `final_summary.txt` and `initialization_summary.json`.

The paper run `args.yaml` records `pretrained: yolo11n.pt`, i.e. native transfer;
this repository's runner reproduces it with `--weights yolo11n.pt
--weights-map none`.

## 4. Evaluation scripts

| Script | Role |
|---|---|
| `eval/eval_ultralytics_common_evaluator.py` | Ultralytics checkpoint -> COCO predictions -> Common Evaluator; source of main-table accuracy |
| `eval/evaluate_coco_predictions_full_metrics.py` | COCOeval wrapper: AP50/75/50-95, AP_small, P/R/F1, ECE, AR300 |
| `eval/export_yolo_split_to_coco.py` | YOLO labels -> frozen COCO GT (with split marker) |
| `eval/benchmark_uniform_latency.py` | uniform FPS benchmark (FP32, b1, 640, warmup 100, iter 500, forward+NMS) |
| `eval/summarize_paper16_unified_common_metrics.py` | Dataset A three-seed summary (archival; paths need editing) |
| `eval/summarize_dataset_b_common_evaluator.py` | Dataset B three-seed report (archival) |
| `eval/paper35_extended_statistics.py` | Bootstrap, IoU localization error, AP75 |
| `eval/paper27_bootstrap_confusion_area.py` | area-stratified AP, confusion matrix, Bootstrap |
| `eval/supplement_paper32_ap75.py` | AP75 supplement |
| `eval/build_paper_mean_sd_tables.py` | mean+/-SD table helper |
| `eval/validate_rerun_configs.py` | rerun-config validation |

Common Evaluator: 640x640; conf=0.001; class-aware NMS IoU=0.70; max_det=300;
COCO 101-point AP; P/R at class-mean-F1 max; A `--area-scale 4`, B `--area-scale
1`; outputs `overall`, `per_class`, `quality`.

FPS: CUDA FP32, b1, 640; warm-up 100, timing 500; sync before/after; forward +
decode + class-aware NMS (conf 0.25, IoU 0.70, max_det 300); excludes image
loading/CPU-GPU transfer/preprocessing; mean+/-SD over three seed checkpoints.

## 5. Data build/audit scripts

| Script | Dataset | Role |
|---|---|---|
| `eval/build_lichi_dataset_b_v4.py` | B | clean, group, 70/15/15 split, manifests/audit |
| `eval/audit_lichi_dataset_b_v4.py` | B | independent audit (leakage, hashes, group constraints) |
| `eval/augment_lichi_dataset_b_v4.py` | B | train-only three-category augmentation |
| `eval/verify_lichi_dataset_b_v4.py` | B | independent verification (PASS) |
| `eval/audit_dataset_a_leakage.py` | A | 878 vs 842, exact/perceptual/sequential leakage audit |
| - | A | original build script not preserved; three manifests are authoritative |

## 6. Results files

| File | Content |
|---|---|
| `results/dataset_a_per_seed.csv` | A per-seed accuracy + Common Eval JSON + checkpoint paths |
| `results/dataset_a_three_seed_mean_sd.csv` | A mean+/-SD |
| `results/paper_dataset_b_...__main_results.json/.md` | B per-seed/summary/paired deltas |
| `results/ap75_per_seed.csv`, `ap75_three_seed.csv` | AP75 |
| `results/bootstrap_ci_extended.csv/.json` | Bootstrap 95% CI, IoU localization error |
| `results/dataset_a_area_ap.csv`, `dataset_a_results.json`, `dataset_a_confusion_matrices.csv` | area-stratified AP, confusion matrix |
| `results/common_evaluator/dataset_{a,b}/...` | A/B x two models x 3 seeds Common Evaluator JSON |
| `results/latency/` | FPS raw/summary CSV |
| `results/Appendix_C/` | Supplementary's 13 per-seed tables |
| `results/paper_3.12_experiment_summary_zh.md` | original Chinese ledger |

## 7. Code-change boundary

`src/ultralytics` is a **modified copy** of Ultralytics 8.4.11. Relative to
upstream:

1. added `advanced.py::SPOConv` (FSDConv);
2. added `small_object_modules.py::SHAB` (SHEB) and controls (ASPP, DilatedConv,
   RepConvBlock, MultiScaleDWConv, ReceptiveFieldBlock, CFSB);
3. exported them in `modules/__init__.py`;
4. imported/parsed them in `tasks.py`;
5. added FSDH/ablation/control YAMLs under `cfg/models/11/`;
6. added paper-specific training/evaluation wrappers under `train/` and `eval/`.

Use this repository's `src`, not a pip-installed upstream package:

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
python train/origin_train.py ...
```