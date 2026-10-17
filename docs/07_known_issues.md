# 07. Known issues, protocol caveats and usage boundaries

## 1. One last-digit issue remains in paper 3.23

**Appendix B.2, SPDConv+SHEB row, mAP50-95 SD:**

- 3.23 docx: `66.649+/-0.334`
- per-seed recomputation: `66.649+/-0.335`
- paper 3.12 ledger: `66.649+/-0.335` (correct)

Per-seed: 66.701 / 66.291 / 66.954; mean = 66.648666...; sample SD (ddof=1)
= 0.334584... -> **0.335**.

The row label was fixed in 3.23 from the old `FSDConv+SHEB` to `SPDConv+SHEB`;
only the SD remains. Suggested action: change the cell to `+/-0.335`.

## 2. Dataset A annotation lacks `info.split`

The original file:

```text
external_configs/rtdetr_coco_annotations/instances_test.json
```

has only `info.description`, no `info.split`. The current
`scripts/eval_ultralytics_common_evaluator.py` checks:

```python
if ground_truth.get("info", {}).get("split") != "test":
    raise ValueError("Common Evaluator requires a test annotation file")
```

so it rejects the original file. This repository's
`annotations/dataset_a_test_instances.json` adds:

```json
"info": {
  "description": "Lychee three-category test split",
  "split": "test",
  "original_file": "external_configs/rtdetr_coco_annotations/instances_test.json",
  "original_sha256": "..."
}
```

All other fields (126 images / 2,775 annotations / categories 0/1/2) are
unchanged. The reproduction matches the paper exactly, so the fix does not alter
ground truth. Dataset B already had `info.split="test"` and is unchanged.

## 3. Fix status of 3.23 relative to 3.13

The 3.12 ledger's Section 5 targets **3.13**. Current 3.23:

| 3.13 issue | 3.23 status |
|---|---|
| Table 3 SHEB FPS 122.48+/-1.80 | fixed to **122.01+/-1.17** |
| Table 8 ripe AP50-95 Delta +4.739 | fixed to **+4.740** |
| Table 13 green AP50-95 +1.530 | fixed to **+1.529** |
| Table 13 red AP50 -0.384 | fixed to **-0.385** |
| Table 13 young AP50 +0.998 | fixed to **+0.999** |
| YOLOv7-tiny mAP50-95 66.343 | fixed to **66.341** |
| Section 3.8 AP75 0.283 | fixed to **0.282** |
| Appendix B.2 label `FSDConv+SHEB` | fixed to **SPDConv+SHEB** |
| Appendix B.2 SD 0.334 | **still 0.334; should be 0.335** |
| Two shifted appendix per-seed tables | 3.23 no longer contains them; per-seed data is in Supplementary Appendix C |
| "Table X" placeholder | not detected in 3.23 |

Conclusion: 3.23 has fixed the major issues; only B.2's SD last digit remains.

## 4. Hard-coded paths in archival scripts

These are **archived campaign scripts**, not turnkey:

| Script | Issue |
|---|---|
| `train/run_all_models.ps1` | Dataset B; original absolute paths for dataset/weights/python; provenance checks require raw data + audit files |
| `train/run_all_train_val.ps1` | Dataset B legacy campaign; absolute paths |
| `eval/summarize_paper16_unified_common_metrics.py` | `TRAIN_ROOT`/`QUALITY_ROOT` point to the original workspace |
| `eval/summarize_dataset_b_common_evaluator.py` | default paths/model names from the original campaign |
| `eval/paper35_extended_statistics.py` | original input paths |
| `eval/paper27_bootstrap_confusion_area.py` | original input paths |
| `eval/build_lichi_dataset_b_v4.py` etc. | original data-build paths/protocol files |

Recommended entry: `train/run_reproduction.py` (A/B, sets `PYTHONPATH`,
generates a runtime YAML, relative/parameterized paths). The other scripts are
kept as method/protocol references.

## 5. `paper_train_single.py` is not the final paper entry

The original workspace also has `paper_train_single.py`, which:

- hard-codes `mytomato_original.yaml`;
- defaults to `pretrained=False`;
- is not the Dataset A/B paper runner.

This repository does **not** use it as the main entry; `origin_train.py` is the
paper training script. Do not use `paper_train_single.py` to reproduce FSDH-YOLO.

## 6. Dataset A original build script was not preserved

The original workspace preserved:

- `data/dataset_a/build_summary.json`
- `group_manifest.csv`
- `split_manifest.csv`
- `group_constraint_edges.csv`
- `augmentation_log.csv` / `augmentation_scheme.txt`
- `eval/audit_dataset_a_leakage.py`

but not the script that generated `lychee_groupaware_clean_raw_v2`. **The fixed
split is authoritative via the three manifests**; `build_summary.json` records
thresholds, group counts, constraint edges and zero violations. To rebuild from
the 878 source images, implement the rules in `docs/03_dataset_protocol.md`, or
restore the split directly from `split_manifest.csv` (recommended; it guarantees
per-file consistency). Dataset B's build/audit/augment/verify scripts are
included under `eval/`.

## 7. Metric and data caveats

1. **A/B AP_small differ**:
   - A: `AP_small(640)` = original-image `[0,4096) px2` (`--area-scale 4`);
   - B: `AP_small(COCO)` = original-image `<1024 px2` (`--area-scale 1`).
   Do not compare the two numerically.
2. **COCO AP_s vs AP_small(640)**: A's COCO AP_s has only 37 GT and is unstable;
   the 3.12 ledger notes its delta CI crosses zero.
3. **P/R operating point**: confidence maximizing class-mean F1, not fixed
   conf=0.25.
4. **Confusion matrix**: conf 0.25, IoU 0.50, row-normalized, Miss/FN column;
   a different operating point from P/R/AP.
5. **FPS**: forward+decode+NMS (conf 0.25), excludes image loading/preprocessing;
   not end-to-end.
6. **Params**: fused inference-time values, not state-dict.
7. **Native val vs Common Evaluator**: paper tables use the latter; best.pt
   selection uses the former.
8. **Dataset B class IDs are not ordinal**: `young=3`; maturity order is
   `[young, green, half, red]` and targets are `[1,2,3,0]`.
9. **Dataset B is not an independent domain test set**: `scope_note` says
   "same-source leakage-repaired group-aware split"; Section 3.8 conclusions
   should be limited to cross-dataset/cross-label-system, not cross-orchard or
   cross-camera.

## 8. Checkpoint and model naming

1. **A and B heads differ**: A 3-class, B 4-class; do not interchange.
2. **`FSDH-YOLO_n` vs `FSDH-YOLO`**: paper Dataset A uses `FSDH-YOLO`; paper
   Table 12 writes `FSDH-YOLO` for B, while this repository uses `FSDH-YOLO_n`
   to mark the n-scale 4-class model. Same architecture.
3. **`SHEB` vs `SHAB`**: paper uses SHEB; code/checkpoints use SHAB. Same module.
4. **`FSDConv` vs `SPOConv`**: paper uses FSDConv; implementation class is
   `SPOConv` for checkpoint compatibility.
5. **`last.pt` is not the paper checkpoint**: paper uses `best.pt`.
6. **Checkpoints are AGPL-3.0**: keep `LICENSE`/`NOTICE` on redistribution.

## 9. Repository/upload notes

1. `src/ultralytics` is **modified**; do not overwrite it with a pip-installed
   upstream package. Set `PYTHONPATH=<repo>/src`.
2. Checkpoints total about 74 MB; every file is under 7 MB; ordinary Git is
   sufficient (no LFS required).
3. Dataset images are not uploaded; `data/` contains manifests/logs/audits only.
4. `runs/` and `out/` are gitignored; training outputs are not uploaded.
5. `results/Appendix_C/` is MIT-licensed from the Supplementary repository; its
   license copy is retained.
6. Android/ONNX/TFLite are not in this repository; use the Supplementary
   `android_deployment/`.