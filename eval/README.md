# Evaluation scripts

## Main evaluator

```bash
PYTHONPATH=src python eval/eval_ultralytics_common_evaluator.py \
  --model weights/dataset_a/FSDH-YOLO_n_seed0_best.pt \
  --model-name FSDH-YOLO --seed 0 \
  --images /path/to/lychee_groupaware_clean_aug_v2/images/test \
  --annotations annotations/dataset_a_test_instances.json \
  --out out/metrics.json --predictions-out out/preds.json \
  --imgsz 640 --device 0 --conf 0.001 --iou 0.7 --max-det 300 \
  --bins 15 --area-scale 4
```

- Dataset A: `--area-scale 4`; Dataset B: `--area-scale 1`.
- Outputs `overall`, `per_class`, `quality`.
- Depends on `evaluate_coco_predictions_full_metrics.py` (same directory) and
  `pycocotools`.

## Data export

```bash
python eval/export_yolo_split_to_coco.py \
  --data /path/to/data.yaml --split test --out annotations/test.json
```

## Latency

```bash
PYTHONPATH=src python eval/benchmark_uniform_latency.py \
  --manifest latency_manifest.csv --image /path/to/test.jpg --out out/latency \
  --imgsz 640 --warmup 100 --iterations 500
```

Manifest columns: `section,model,seed,weights,imgsz`.

## Statistics / summaries

- `summarize_paper16_unified_common_metrics.py`: Dataset A three-seed summary (archival paths).
- `summarize_dataset_b_common_evaluator.py`: Dataset B report (archival paths).
- `paper35_extended_statistics.py`: Bootstrap, IoU localization error, AP75.
- `paper27_bootstrap_confusion_area.py`: area-stratified AP, confusion matrix, Bootstrap.
- `supplement_paper32_ap75.py`: AP75 supplement.
- `build_paper_mean_sd_tables.py`: mean+/-SD helper.

> Summary scripts still contain original-workspace absolute paths; update the
> top constants or read `runs/*/common_eval/**/seed*__test_metrics_common.json`.

## Data build/audit

- `build_lichi_dataset_b_v4.py`
- `audit_lichi_dataset_b_v4.py`
- `augment_lichi_dataset_b_v4.py`
- `verify_lichi_dataset_b_v4.py`
- `audit_dataset_a_leakage.py`
- `validate_rerun_configs.py`

## Protocol

See `docs/06_reproduction.md` section 3.2 and
`docs/08_source_inventory.md` section 4.