# Training scripts

## Recommended entry

```bash
python train/run_reproduction.py --dataset a --dataset-root /path/to/lychee_groupaware_clean_aug_v2
python train/run_reproduction.py --dataset b --dataset-root /path/to/lichi_701515_groupaware_clean_v4_aug
```

- sets `PYTHONPATH=<repo>/src`;
- generates a runtime YAML (or pass `--data-yaml`);
- runs train -> native val -> Common Evaluator;
- `--plan-only`, `--skip-train`, `--latency`, `--models`, `--seeds`.

## Paper entry

`origin_train.py`:

```bash
PYTHONPATH=src python train/origin_train.py \
  --model configs/yolo11n_FSDConv-SHAB.yaml \
  --weights weights/pretrained/yolo11n.pt --weights-map none \
  --data /path/to/data.yaml \
  --project runs/a/train --name fsdh_seed0 \
  --epochs 300 --batch 12 --imgsz 640 --workers 6 --val-workers 2 \
  --patience 100 --seed 0 --fraction 1.0 --iou-type ciou
```

Key behavior:

- calls `init_seeds(seed+1, deterministic=True)` before model construction;
- `--weights-map none` = native same-name/same-shape transfer (paper run init);
- training device is hard-coded to `"0"`;
- writes `weights/best.pt`, `weights/last.pt`, `results.csv`, `final_summary.txt`.

## Archived campaigns

- `run_all_models.ps1`: Dataset B original campaign (absolute paths + provenance checks).
- `run_all_train_val.ps1`: Dataset B legacy campaign (absolute paths).

These are historical records; use `run_reproduction.py` for new runs.

## Validation

- `val.py`: native Ultralytics val/test, writes `metrics_summary.json`.
  Paper table values use Common Evaluator, not this file.
- `save_yolo_final_summary.py`: writes `final_summary.txt`.

## Notes

- Do not overwrite `src/ultralytics` with a pip-installed upstream package.
- Training requires a CUDA GPU (device "0").
- Early stopping changes actual epochs; use `best.pt`.