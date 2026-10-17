# FSDH-YOLO Reproduction Materials

**FSDH-YOLO: Feature-preserving Space-to-Depth and Hierarchical-enhancement YOLO**

Companion repository for the manuscript:

> *A Spatial-to-Depth Reorganization and Hierarchical Enhancement Network for
> Litchi Maturity Detection in Natural Orchard Environments*
>
> Dehua Chen, Haorong Wu, Xiaoxiao Li, Hao Teng, Junjie Yang, Zihan Teng, Yang Liu
> School of Electronic Information and Electrical Engineering, Chengdu University

This is the **training and evaluation side** of the reproducibility package. It
complements the existing
[FSDH-YOLO-Supplementary](https://github.com/ther608/FSDH-YOLO-Supplementary)
repository:

| Repository | Contents |
|---|---|
| **This repository** (`FSDH-YOLO-Reproduction`) | Modified Ultralytics source, model configs, training/evaluation scripts, fixed dataset splits and cleaning/grouping manifests, frozen COCO annotations, 12 `best.pt` checkpoints, paper results and per-seed CSVs |
| [FSDH-YOLO-Supplementary](https://github.com/ther608/FSDH-YOLO-Supplementary) | Appendix C per-seed tables, evaluation protocol, Android deployment app, ONNX models, raw on-device measurements |

> **Dataset images are not redistributed here.** Both datasets are public. This
> repository provides the fixed split/cleaning/grouping manifests, checksums and
> rules needed to reconstruct the paper's splits from the original sources.

### Provided / not provided

| Provided | Not provided |
|---|---|
| Modified Ultralytics 8.4.11 source (`src/`) | Dataset images (public sources; download separately) |
| 22 model / ablation / control YAMLs (`configs/`) | Android project and ONNX/TFLite models (in the Supplementary repository) |
| Paper training, evaluation and data-build scripts (`train/`, `eval/`) | All 656 run checkpoints (this repo keeps the main model + baseline x A/B x 3 seeds) |
| Fixed split / cleaning / grouping / augmentation manifests for both datasets (`data/`) | External-framework (YOLOv7/RT-DETR) weights |
| Frozen test-set COCO annotations (`annotations/`) | Manuscript Word/LaTeX source |
| 12 paper `best.pt` checkpoints + `yolo11n.pt` (`weights/`) | - |
| Paper results, per-seed CSVs, Common Evaluator JSONs, Appendix C (`results/`) | - |
| Coverage audit, dataset protocol, weight inventory and reproduction docs (`docs/`) | - |

The repository follows the organisation of the existing Supplementary repository:
root README + `CITATION.cff` + `LICENSE`/`LICENSE-docs` + `.gitattributes` +
per-directory READMEs + `PROVENANCE.md` + machine-readable manifests. The
difference is that this repository adds the **training code, PyTorch checkpoints
and fixed split manifests**.

---

## 1. Headline results (paper protocol, mean +/- sample SD over 3 seeds)

### Dataset A (primary, 3 classes)

| Model | P/% | R/% | mAP50/% | mAP50-95/% | Params/M | FPS |
|---|---|---|---|---|---|---|
| YOLO11n | 78.007+/-1.588 | 78.174+/-2.110 | 78.984+/-0.959 | 64.429+/-0.603 | 2.583 | 146.69+/-4.87 |
| **FSDH-YOLO** | **79.989+/-2.168** | 78.443+/-1.127 | **81.766+/-0.994** | **67.297+/-0.718** | **2.997** | 123.39+/-2.94 |

### Dataset B (cross-dataset, 4 native classes)

| Model | P/% | R/% | mAP50/% | mAP50-95/% | AP75/% | AP_small(640)/% |
|---|---|---|---|---|---|---|
| YOLO11n | 81.097+/-1.471 | 82.504+/-2.222 | 84.902+/-0.748 | 61.238+/-0.374 | 70.577+/-0.443 | 32.332+/-1.812 |
| FSDH-YOLO-n | 81.073+/-2.128 | 82.820+/-0.674 | 84.492+/-0.707 | 60.210+/-0.621 | 70.294+/-0.630 | 32.524+/-1.329 |

> The manuscript labels the Dataset B model `FSDH-YOLO`; this repository uses
> `FSDH-YOLO_n` in file names to distinguish it from the Dataset A configuration.
> Both refer to the same architecture (YOLO11n-scale) with a 4-class head.

**Exact re-evaluation performed with the bundled weights and Common Evaluator
(640x640):**

| Dataset | Model | seed | mAP50 | mAP50-95 | AP75 | AP_small |
|---|---|---|---|---|---|---|
| A | FSDH-YOLO | 0 | 80.931% | 66.670% | 76.193% | 57.577% |
| B | FSDH-YOLO-n | 0 | 83.993% | 59.683% | 69.845% | 32.795% |

Both rows reproduce the paper's seed-0 values to the last printed digit. See
`docs/05_weight_inventory.md` and `docs/06_reproduction.md`.

---

## 2. Repository layout

```text
FSDH-YOLO-Reproduction/
|-- README.md
|-- LICENSE                         # AGPL-3.0 (modified Ultralytics source)
|-- LICENSE-docs                    # CC BY 4.0 for docs/manifests/results
|-- NOTICE
|-- CITATION.cff
|-- PROVENANCE.md                   # artifact -> source run -> SHA256
|-- CHECKSUMS.sha256                # repository-wide checksums
|-- .gitattributes
|-- requirements.txt
|-- src/ultralytics/                # modified Ultralytics 8.4.11 (self-contained)
|-- configs/                        # 22 FSDH/ablation/control YAMLs
|-- train/                          # origin_train.py, val.py, run_reproduction.py, archived PS1 campaigns
|-- eval/                           # Common Evaluator, latency, statistics, data build/audit
|-- data/dataset_a/                 # fixed split, group manifest, constraint edges, augmentation log
|-- data/dataset_b/                 # fixed split, group manifest, cleaning/audit, augmentation
|-- annotations/                    # frozen test-set COCO annotations
|-- weights/                        # 12 best.pt + yolo11n.pt + MANIFEST.csv + SHA256SUMS
|-- results/                        # paper tables, per-seed CSV, Common Evaluator JSON, Appendix C
`-- docs/                           # 01-09 audit/protocol/weights/reproduction/source/results docs
```

Key directories:

| Path | Purpose |
|---|---|
| `src/ultralytics/` | Self-contained modified Ultralytics 8.4.11; `SPOConv` (FSDConv) and `SHAB` (SHEB) live here |
| `configs/` | Main FSDH-YOLO YAML + ablation/branch/SPD/control YAMLs |
| `train/` | Paper training entry, native val, unified A/B runner |
| `eval/` | Framework-neutral Common Evaluator, latency benchmark, statistical and data-build scripts |
| `data/` | Per-file fixed split, group assignment, cleaning/grouping edges, augmentation logs |
| `annotations/` | Frozen COCO test annotations (A: 126/2,775; B: 212/3,263) |
| `weights/` | 12 paper `best.pt` + pretrained `yolo11n.pt` + manifests |
| `results/` | Per-seed CSV, mean/SD, Common Evaluator JSON, Appendix C, latency |
| `docs/` | Coverage audit, existing repos, dataset protocol, group summary, weights, reproduction, known issues, source inventory, results index |

---

## 3. Quick start

### 3.1 Environment

Paper environment: Windows 11 64-bit, Intel i5-14400F, NVIDIA RTX 5060 Ti
16 GB, Python 3.10, CUDA 12.8, PyTorch 2.10+cu128, Ultralytics 8.4.11.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`src/ultralytics` **is** the modified 8.4.11 source. Set
`PYTHONPATH=<repo>/src` when training/evaluating (`run_reproduction.py` does it
automatically).

### 3.2 Data

Download the public datasets and arrange them as:

```text
<Dataset A root>/
|-- images/{train,val,test}/
|-- labels/{train,val,test}/
`-- lychee_groupaware_clean_aug_v2.yaml

<Dataset B root>/
|-- train/{images,labels}/
|-- val/{images,labels}/
|-- test/{images,labels}/
`-- lichi_maturity_groupaware_clean_v4_aug.yaml
```

`data/dataset_a/` and `data/dataset_b/` contain the **per-file fixed split and
group assignment** (`split_manifest.csv`, `group_manifest.csv`). Reconstruct the
splits by filename and SHA256; see `docs/03_dataset_protocol.md`.

### 3.3 Train + validate + Common Evaluator

```bash
# Dataset A: YOLO11n and FSDH-YOLO, seeds 0/1/2
python train/run_reproduction.py \
  --dataset a \
  --dataset-root /path/to/lychee_groupaware_clean_aug_v2

# Dataset B: YOLO11n and FSDH-YOLO-n, seeds 0/1/2
python train/run_reproduction.py \
  --dataset b \
  --dataset-root /path/to/lichi_701515_groupaware_clean_v4_aug
```

Inspect commands without running:

```bash
python train/run_reproduction.py --dataset a --dataset-root ... --plan-only
```

Evaluate existing checkpoints only (no training):

```bash
python train/run_reproduction.py --dataset a --dataset-root ... --skip-train
```

Outputs:

```text
runs/<a|b>/train/<run_name>/weights/best.pt
runs/<a|b>/test/<run_name>_test/metrics_summary.json
runs/<a|b>/common_eval/<model>/seed<N>__test_metrics_common.json
runs/<a|b>/common_eval/<model>/seed<N>__test_predictions_coco.json
```

Common Evaluator protocol: 640x640 square inference, `conf=0.001`, class-aware
NMS `IoU=0.70`, `max_det=300`, COCO 101-point AP. Dataset A uses
`AP_small(640)` with an original-image area threshold of `[0, 4096) px2`
(`--area-scale 4`); Dataset B uses `<1024 px2` (`--area-scale 1`). See
`docs/06_reproduction.md`.

### 3.4 Re-evaluate bundled checkpoints (no training)

```bash
PYTHONPATH=src python eval/eval_ultralytics_common_evaluator.py \
  --model weights/dataset_a/FSDH-YOLO_n_seed0_best.pt \
  --model-name FSDH-YOLO --seed 0 \
  --images /path/to/lychee_groupaware_clean_aug_v2/images/test \
  --annotations annotations/dataset_a_test_instances.json \
  --out out/fsdh_a_seed0_metrics.json \
  --predictions-out out/fsdh_a_seed0_predictions.json \
  --imgsz 640 --device 0 --conf 0.001 --iou 0.7 --max-det 300 --area-scale 4
```

---

## 4. Checkpoints

`weights/MANIFEST.csv` lists, for every file:

- dataset, model, seed;
- corresponding config;
- fused parameter count (paper protocol) and state-dict parameter count;
- best epoch (selected by validation mAP50-95);
- paper Common Evaluator mAP50 / mAP50-95;
- file size and SHA256;
- original source run.

There are 12 trained checkpoints (A/B x {FSDH-YOLO, YOLO11n} x 3 seeds) plus
`yolo11n.pt`. All were loaded with `YOLO(...)`; A has `nc=3`, B has `nc=4`.

---

## 5. Coverage audit of the paper ledger

`docs/01_coverage_audit.md` compares
the original paper 3.12 experiment ledger (Chinese text, retained as
`results/paper_3.12_experiment_summary_zh.md`) with the current paper 3.23
CEA manuscript (docx). Summary:

- Main-text Tables 3-8 and 10-13, plus Appendix B.1-B.4, are numerically
  **almost fully covered**;
- **Not covered**: Dataset A construction/cleaning/grouping, Table 1
  augmentation ranges, Table 2 training hyperparameters, Table 9 qualitative
  examples, Table 14 / Appendix B.5 Android deployment, and a file-level
  checkpoint inventory;
- Version 3.23 has fixed most issues that the 3.12 ledger flagged against 3.13;
- One last-digit discrepancy remains: Appendix B.2 `SPDConv+SHEB` prints
  `66.649+/-0.334`; the per-seed recomputation is `66.649+/-0.335` (the 3.12
  ledger also has 0.335).

---

## 6. Data sources and license

- **Dataset A**: Zhang et al., *An RGB-D image dataset for lychee detection and
  maturity classification for robotic harvesting*, arXiv:2510.16800
  (`doi:10.48550/arXiv.2510.16800`). This work uses the RGB subset: 36 indoor
  tabletop images removed, 842 outdoor orchard images retained.
- **Dataset B**: Zhao, *lichi-maturity*, Mendeley Data V1,
  `doi:10.17632/c3rk9gv4w9.1`.

Code and trained checkpoints are released under **AGPL-3.0** (they include a
modified Ultralytics 8.4.11 source tree). Documentation, split/cleaning
manifests, annotations and results are released under **CC BY 4.0**
(`LICENSE-docs`). `results/Appendix_C/` is MIT-licensed, copied from the
Supplementary repository; see `LICENSES/FSDH-YOLO-Supplementary-MIT.txt`.
Dataset images remain under their original licenses.

---

## 7. Citation

```bibtex
@article{chen2026fsdh,
  title   = {A Spatial-to-Depth Reorganization and Hierarchical Enhancement Network
             for Litchi Maturity Detection in Natural Orchard Environments},
  author  = {Chen, Dehua and Wu, Haorong and Li, Xiaoxiao and Teng, Hao and
             Yang, Junjie and Teng, Zihan and Liu, Yang},
  year    = {2026},
  note    = {Manuscript}
}
```

If you use the code or checkpoints, please also cite Ultralytics YOLO11.
