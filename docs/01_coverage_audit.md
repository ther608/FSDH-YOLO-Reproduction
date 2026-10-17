# 01. Coverage audit: `paper 3.12` ledger vs `paper 3.23` docx

Audit date: 2026-10-17

Objects:

- **Ledger (md)**: the paper 3.12 experiment ledger (Chinese text), retained at `results/paper_3.12_experiment_summary_zh.md`
  (765 lines; described as a data summary, not a manuscript edit; its Section 5
  explicitly checks **paper 3.13**, not 3.23). A copy with an English header is
  retained at `results/paper_3.12_experiment_summary_zh.md`.
- **Manuscript (docx)**: paper 3.23 CEA manuscript (Chinese filename)
  (after extraction: 513 paragraphs/table rows, about 122,368 characters;
  14 numbered main-text tables + 5 Appendix B tables).

## 0. One-paragraph verdict

**The 3.12 ledger is a reliable "numerical ledger" for the main results, but it
is not a complete reproducibility package.**

- Main-text Tables 3-8 and 10-13, plus Appendix B.1-B.4: numerically covered,
  with one last-digit exception in B.2.
- Dataset construction, cleaning and grouping: only partially covered (Dataset B
  split numbers); Dataset A is essentially absent.
- Training configuration, deployment experiments, qualitative analysis and
  confusion-matrix values: not covered.
- Checkpoints: only path patterns and a few concrete paths, no file-level
  inventory (size, SHA256, best epoch, source run).

Therefore: **use the ledger to check paper tables, not to reconstruct splits or
collect weights.** This repository fills those gaps.

---

## 1. Coverage matrix

| Content in paper 3.23 | Covered by 3.12 ledger? | Notes |
|---|---|---|
| Dataset A source/filtering: 878 -> remove 36 indoor -> 842 | No | Ledger only mentions 126 test images and test GT counts |
| Dataset A annotation cleaning: IoU>0.90, 936 candidate groups/1,873 boxes, 932 keep-one, 4 keep-all, 933 removed (56/554/323) | No | No cleaning rules/counts; only build path |
| Dataset A grouping: pHash<=10 & gray32 MAE<=0.06; same-batch index gap<=3 & MAE<=0.10; 672 groups, 32 multi-image groups, largest 38 | No | No grouping rules |
| Dataset A fixed split: 590/126/126 images, 11,083/2,690/2,775 GT, 553/58/61 groups | No | No train/val sizes or group counts |
| Dataset A augmentation: 2,360 images/44,151 GT, class counts 15,268/10,372/18,511 | No | Only mentions the `aug_v2` path |
| Dataset B source/filtering: 1,649 -> remove 154 augmented + 10 duplicate boxes -> 1,495 | Partial | Ledger gives 1,071/212/212 and 4,284 augmented; not 1,649/154/10/1,495 |
| Dataset B grouping: 1,188 groups, 49 multi-image, largest 123, 730 constraint edges | Partial | Groups/edges absent; only final split |
| Dataset B fixed split: 1,071/212/212 images, 12,935/3,267/3,263 GT | Partial | Images and per-class test AP present; per-split GT totals absent |
| Dataset B augmentation: 4,284 images/51,728 GT, operation counts | Partial | Only "4,284 after augmentation" |
| Table 1: augmentation parameter ranges | No | Ledger only says three independent variant categories |
| Table 2: training configuration (batch 12, SGD, lr0/lrf, momentum, weight decay, 300 epochs, patience 100) | No | Ledger Section 0 is the **evaluation** protocol; only 3 seeds and best.pt fitness are mentioned |
| Table 3: multi-scale module comparison (8 rows P/R/mAP/Params/FPS) | Yes | Mean+/-SD, per-seed, FPS, Params; SHEB FPS = 122.01+/-1.17 |
| Table 4: ablation (5 rows) | Yes | Includes YOLO11n-PM capacity control |
| Table 5: small-object area stratification | Yes | GT counts, mean+/-SD, per-seed, Bootstrap |
| Table 6: AP75 localization | Yes | 74.441 / 77.195, +2.754 pp |
| Table 7: IoU localization error on common detections | Yes | Three seeds + mean+/-SD + Bootstrap CI |
| Table 8: maturity-class AP50/AP50-95 | Yes | Delta and Bootstrap CI; ripe Delta +4.740 |
| Table 9: qualitative detection examples | No | No case images/numbers; qualitative, not a numerical result |
| Table 10: general detector comparison | Yes | Six comparison models + FSDH |
| Table 11: recent fruit-detection models | Yes | Ledger 1.7 merges Tables 10/11 (11 rows) |
| Table 12: Dataset B overall | Yes | FSDH written as FSDH-YOLO-n |
| Table 13: Dataset B per-class AP50/AP50-95 | Yes | green/half/red/young |
| Table 14: Android fixed-input/continuous-camera deployment | No | No 6.35/7.76 FPS, 157.5/128.8 ms, 3.89-4.13 FPS |
| Appendix B.1: YOLO11 n/s/m scale selection | Yes | Includes Params/FPS |
| Appendix B.2: SPDConv vs FSDConv | Partial, one last-digit mismatch | Ledger SPDConv+SHEB 66.649+/-**0.335**; 3.23 docx 66.649+/-**0.334** |
| Appendix B.3: SHEB branch ablation (21 rows) | Yes | L/M/G/LM/LG/MG/full/FSDConv+SHEB-M |
| Appendix B.4: Bootstrap robustness (18 rows) | Yes | CI + decision column |
| Appendix B.5: three continuous-camera runs + diagnostic rerun | No | No 242.34-257.09 ms, P95, 606 frames |
| Figure 9: confusion-matrix values | Partial | Ledger gives CSV/PNG paths; values are not tabulated |
| Weight inventory: size/SHA256/best epoch | No | Path patterns and `per_seed.csv` checkpoint column only |
| Environment (CPU/GPU/Python/CUDA/PyTorch/Ultralytics) | Partial | Evaluation/latency protocol present; hardware/versions absent |
| Data/code/weight availability statement | No | Internal data summary, not an availability statement |

Legend: Yes = covered; Partial = partially covered; No = not covered.

---

## 2. Numerical cross-check

Method: extract every table from the 3.23 docx, collect tokens matching
`number.number` and `number.number+/-number.number`, and search for each token
in the ledger text.

Results:

- Tables 3-8 and 10-13, plus Appendix B.1-B.4: every numerical token was found
  in the ledger.
- The only "docx has it, ledger does not / differs" cases are:
  1. **Table 2 training config**: `0.937` (momentum) and `0.0005` (weight decay)
     are absent from the ledger;
  2. **Table 14 deployment**: `128.8 / 157.5 / 7.76 / 6.35 / 242.34 / 257.09 /
     4.13` are absent;
  3. **Appendix B.5**: `242.34+/-17.86 / 255.11+/-20.66 / 257.09+/-18.09 /
     284.63 / 305.54 / 292.96 / 4.13 / 3.92 / 3.89` are absent;
  4. **Appendix B.2**: docx has `66.649+/-0.334`, ledger has `66.649+/-0.335`
     (see Section 4).

All other values agree, including the easy-to-misplace ones:

- Table 3 SHEB FPS = 122.01+/-1.17 (3.13 had 122.48+/-1.80; 3.23 fixed);
- Table 8 ripe AP50-95 Delta = +4.740 (3.13 had +4.739; 3.23 fixed);
- Table 13 green AP50-95 Delta = +1.529, red AP50 Delta = -0.385, young AP50
  Delta = +0.999 (3.13 had +1.530/-0.384/+0.998; 3.23 fixed);
- Table 10 YOLOv7-tiny mAP50-95 = 66.341 (3.13 text had 66.343; 3.23 fixed);
- Dataset B AP75 Delta = -0.282 (3.13 text had 0.283; 3.23 fixed).

---

## 3. Version drift: the ledger's Section 5 checks 3.13, not 3.23

The ledger's Section 5 says it extracted 22 experimental tables from
paper 3.13 (docx); 19 matched, one table needed a SHEB FPS correction, and two
appendix per-seed tables were shifted by one row.

Current 3.23:

- The 3.13 issues flagged by the ledger are **essentially fixed** (the five
  items listed in Section 2);
- The two shifted per-seed appendix tables **no longer exist in 3.23**; per-seed
  data moved to the Supplementary repository's Appendix C (copied to
  `results/Appendix_C/`);
- Therefore the "22 tables" mapping is **not applicable to 3.23**. 3.23 has 14
  numbered tables + 5 Appendix B tables; Table 9 is qualitative, Table 14 is
  deployment, and neither is in the ledger's "Chapter 3 results" scope.

---

## 4. One remaining last-digit issue in 3.23

Appendix B.2, fourth row in 3.23:

| Label | P | R | mAP50 | mAP50-95 | Params | FPS |
|---|---|---|---|---|---|---|
| SPDConv+SHEB | 78.058+/-1.128 | 79.519+/-1.138 | 80.773+/-0.241 | **66.649+/-0.334** | 2.997 | 125.45+/-2.16 |

Per-seed values (Appendix C / ledger B.2):

| seed | mAP50-95 |
|---|---|
| 0 | 66.701 |
| 1 | 66.291 |
| 2 | 66.954 |

Recomputation:

- mean = 66.648666... -> 66.649
- sample SD (ddof=1) = 0.334584... -> **0.335**

So 3.23's `0.334` is a last-digit rounding error; the ledger's `0.335` is
correct. The row label was fixed from the old `FSDConv+SHEB` to
`SPDConv+SHEB`; only the SD remains.

> Suggested action: change the cell to `66.649+/-0.335` before submission.
> This matches ledger line 553 and Issue 1 in the Supplementary repository's
> `reproduction_notes`.

---

## 5. How to use this audit

1. **Use the ledger as the numerical ledger**: Tables 3-8, 10-13 and B.1-B.4
   are a reliable intermediate layer for checking 3.23. This repository keeps
   the source files and per-seed CSVs under `results/`.
2. **Do not use the ledger as the dataset protocol**: see
   `docs/03_dataset_protocol.md` and the `data/` manifests for A's
   842->672 groups->590/126/126, B's 1495->1188 groups->1071/212/212, cleaning
   thresholds and augmentation counts.
3. **Do not use the ledger as the checkpoint inventory**: see
   `weights/MANIFEST.csv` and `docs/05_weight_inventory.md` for file-level
   SHA256, best epoch and Common Evaluator metrics.
4. **Use the Supplementary repository for deployment**: Table 14, Appendix B.5
   and raw Android records are in
   `ther608/FSDH-YOLO-Supplementary`; this repository links to them but does not
   copy the Android project.
5. **One manuscript fix remains**: Appendix B.2 `0.334 -> 0.335`.
