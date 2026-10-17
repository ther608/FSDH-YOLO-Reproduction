# 05. FSDH-YOLO checkpoint inventory and verification

## 0. Summary

This repository includes **12 paper checkpoints** (`best.pt`) and **one
pretrained initialization** (`yolo11n.pt`):

- Dataset A: FSDH-YOLO x 3 seeds + YOLO11n x 3 seeds;
- Dataset B: FSDH-YOLO-n x 3 seeds + YOLO11n x 3 seeds;
- All are `best.pt` selected by validation mAP50-95, not `last.pt`;
- SHA256, size, best epoch and metrics are in `weights/MANIFEST.csv` and
  `weights/SHA256SUMS`.

## 1. File list

### 1.1 Dataset A (3-class head, nc=3)

| File | Model | seed | Fused params/M | best epoch | val mAP50-95 | Common Eval mAP50 | Common Eval mAP50-95 | Size |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `dataset_a/FSDH-YOLO_n_seed0_best.pt` | FSDH-YOLO | 0 | 2.997089 | 63 | 0.67730 | 80.931 | 66.670 | 6,331,272 B |
| `dataset_a/FSDH-YOLO_n_seed1_best.pt` | FSDH-YOLO | 1 | 2.997089 | 74 | 0.67704 | 81.500 | 67.140 | 6,332,680 B |
| `dataset_a/FSDH-YOLO_n_seed2_best.pt` | FSDH-YOLO | 2 | 2.997089 | 52 | 0.67829 | 82.866 | 68.080 | 6,329,864 B |
| `dataset_a/YOLO11n_seed0_best.pt` | YOLO11n | 0 | 2.582737 | 41 | 0.65683 | 80.068 | 65.120 | 5,474,330 B |
| `dataset_a/YOLO11n_seed1_best.pt` | YOLO11n | 1 | 2.582737 | 130 | 0.65437 | 78.246 | 64.151 | 5,485,722 B |
| `dataset_a/YOLO11n_seed2_best.pt` | YOLO11n | 2 | 2.582737 | 44 | 0.65103 | 78.637 | 64.016 | 5,474,714 B |

### 1.2 Dataset B (4-class head, nc=4)

| File | Model | seed | Fused params/M | best epoch | val mAP50-95 | Common Eval mAP50 | Common Eval mAP50-95 | Size |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `dataset_b/FSDH-YOLO_n_seed0_best.pt` | FSDH-YOLO-n | 0 | 2.997284 | 82 | 0.64847 | 83.993 | 59.683 | 6,344,712 B |
| `dataset_b/FSDH-YOLO_n_seed1_best.pt` | FSDH-YOLO-n | 1 | 2.997284 | 81 | 0.64979 | 85.301 | 60.895 | 6,344,584 B |
| `dataset_b/FSDH-YOLO_n_seed2_best.pt` | FSDH-YOLO-n | 2 | 2.997284 | 77 | 0.65575 | 84.181 | 60.051 | 6,344,072 B |
| `dataset_b/YOLO11n_seed0_best.pt` | YOLO11n | 0 | 2.582932 | 59 | 0.65167 | 85.729 | 61.466 | 5,487,514 B |
| `dataset_b/YOLO11n_seed1_best.pt` | YOLO11n | 1 | 2.582932 | 74 | 0.65228 | 84.706 | 61.442 | 5,489,434 B |
| `dataset_b/YOLO11n_seed2_best.pt` | YOLO11n | 2 | 2.582932 | 52 | 0.64837 | 84.272 | 60.807 | 5,486,618 B |

### 1.3 Pretrained initialization

| File | Purpose | Size |
|---|---|---:|
| `pretrained/yolo11n.pt` | YOLO11n baseline init; FSDH `--weights` | 5,613,764 B |

> FSDH training uses `--weights pretrained/yolo11n.pt --weights-map none`,
> i.e. Ultralytics native same-name/same-shape transfer; FSDConv/SHEB parameters
> are initialized by the framework from the seed. Do not use
> `--weights-map fsdh_yolo11n` for the paper runs; that is a semantic-layer
> mapping option, not the reported-run default.

## 2. Parameter-count convention

Paper Tables 3-11 use **fused inference-time** parameters (Ultralytics
`summary (fused)`):

| Configuration | Fused/M | State-dict/M | Note |
|---|---:|---:|---|
| Dataset A FSDH-YOLO | 2.997089 | 3.020234 | 3-class head |
| Dataset A YOLO11n | 2.582737 | 2.605882 | 3-class head |
| Dataset B FSDH-YOLO-n | 2.997284 | 3.020429 | 4-class head |
| Dataset B YOLO11n | 2.582932 | 2.606077 | 4-class head |

- Fused: Conv+BN merged, training-only parameters removed; matches paper
  `2.997 M` / `2.583 M`.
- State-dict: includes BN and other tensors; do not compare with paper tables.
- Dataset B is 195 parameters larger than A due to `nc=3 -> 4` in the head.

## 3. Checkpoint-to-table mapping

| Paper table | Checkpoints used | Notes |
|---|---|---|
| Table 3 multi-scale module comparison | Dataset A module `best.pt` (this repo: FSDH/YOLO11n only) | Other module checkpoints are not bundled; results are in `results/` |
| Table 4 ablation | A: YOLO11n / FSDConv / SHEB / FSDH-YOLO | This repo bundles YOLO11n and FSDH; single-module checkpoints remain in the original workspace |
| Table 5 small object | A: YOLO11n / FSDH-YOLO | Bundled |
| Table 6 AP75 | A: YOLO11n / FSDH-YOLO | Bundled |
| Table 7 box error | A: YOLO11n / FSDH-YOLO | Bundled |
| Table 8 class AP | A: YOLO11n / FSDH-YOLO | Bundled |
| Table 10 general models | A: YOLO11n + external models | Only YOLO11n/FSDH bundled; external weights in original workspace |
| Table 11 recent models | A: reproduced models | Reproduced weights not bundled |
| Table 12/13 Dataset B | B: YOLO11n / FSDH-YOLO-n | Bundled |
| Appendix B.1 scale selection | A: YOLO11n/s/m | Only n bundled |
| Appendix B.2 SPDConv/FSDConv | A: SPDConv / FSDConv / SPDConv+SHEB / FSDH | Only FSDH bundled; SPD weights in original workspace |
| Appendix B.3 SHEB branches | A: L/M/G/LM/LG/MG/full/FSDConv+SHEB-M | Only full FSDH bundled |

> This repository intentionally keeps a **minimal reproducible set**: main model
> + baseline + both datasets x three seeds. It does not upload all 656
> `best.pt` files. Other ablation/control checkpoints can be found from the
> `checkpoint` column of `results/dataset_a_per_seed.csv` in the original
> workspace.

## 4. Verification

### 4.1 Loading

Loaded with PyTorch 2.10+cu128:

| Checkpoint | task | nc | names | Result |
|---|---|---:|---|---|
| A FSDH seed0 | detect | 3 | `{0: unripe, 1: semi-ripe, 2: ripe}` | OK |
| A YOLO11n seed0 | detect | 3 | same | OK |
| B FSDH seed0 | detect | 4 | `{0: green, 1: half, 2: red, 3: young}` | OK |

Checkpoint metadata: Ultralytics 8.4.11; creation date 2026-08-18 (A) /
2026-09-21 (B); optimizer stripped; about 6.3 MB (FSDH) / 5.5 MB (YOLO11n).

### 4.2 Common Evaluator reproduction

Using the bundled checkpoints, annotations and
`eval/eval_ultralytics_common_evaluator.py`:

**Dataset A FSDH seed0:**

| Metric | Reproduced | Source/paper |
|---|---:|---:|
| mAP50 | 80.9311% | 80.931% |
| mAP50-95 | 66.6705% | 66.670% |
| AP75 | 76.1929% | 76.193% |
| AP_small(640) | 57.5771% | 57.577% |
| P / R | 78.581% / 78.364% | 78.581 / 78.364 |

**Dataset B FSDH-n seed0:**

| Metric | Reproduced | Source/paper |
|---|---:|---:|
| mAP50 | 83.9933% | 83.993% |
| mAP50-95 | 59.6828% | 59.683% |
| AP75 | 69.8451% | 69.845% |
| AP_small(COCO) | 32.7949% | 32.795% |
| P / R | 79.807% / 83.488% | 79.807 / 83.488 |

Conclusion: **checkpoints, source, annotations and evaluator agree**; this is a
recomputation to paper values, not merely a file-existence check.

## 5. Usage notes

1. **Do not mix A and B checkpoints**: A is a 3-class head, B is a 4-class head;
   class IDs also differ.
2. **Do not use `last.pt` instead of `best.pt`**: all paper tables use `best.pt`
   (highest validation mAP50-95).
3. **Do not use native Ultralytics val instead of Common Evaluator**: for the
   same checkpoint, native val differs from paper values (e.g. A FSDH seed0
   native val mAP50=0.845 / mAP50-95=0.677; paper 0.809 / 0.667).
4. **Use fused parameters**: `2.997 M`, not the state-dict `3.020 M`.
5. **License**: AGPL-3.0, same as the source; keep `LICENSE`/`NOTICE` on
   redistribution.