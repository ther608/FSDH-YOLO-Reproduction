# Checkpoints

## Contents

- `dataset_a/FSDH-YOLO_n_seed{0,1,2}_best.pt`: Dataset A, FSDH-YOLO, 3 classes, 3 seeds
- `dataset_a/YOLO11n_seed{0,1,2}_best.pt`: Dataset A, YOLO11n baseline, 3 classes, 3 seeds
- `dataset_b/FSDH-YOLO_n_seed{0,1,2}_best.pt`: Dataset B, FSDH-YOLO-n, 4 classes, 3 seeds
- `dataset_b/YOLO11n_seed{0,1,2}_best.pt`: Dataset B, YOLO11n baseline, 4 classes, 3 seeds
- `pretrained/yolo11n.pt`: Ultralytics YOLO11n pretrained initialization
- `MANIFEST.csv`: per-file parameters, best epoch, validation/Common Evaluator metrics, size, SHA256, source run
- `SHA256SUMS`: checksum list

## Conventions

- All are paper `best.pt` (selected by validation mAP50-95), not `last.pt`.
- `params_fused_M` is the paper's fused inference parameter count;
  `params_state_dict_M` is the checkpoint state-dict count; they differ.
- `common_eval_mAP50_percent` / `..._95_percent` are Common Evaluator test-set
  values, not native Ultralytics val.
- Dataset A and B heads differ (3 vs 4 classes); checkpoints cannot be swapped.

## Verification

The bundled A/B seed0 checkpoints were recomputed with
`eval/eval_ultralytics_common_evaluator.py` and match the paper:

- A FSDH seed0: mAP50 80.931%, mAP50-95 66.670%, AP75 76.193%, AP_small 57.577%;
- B FSDH-n seed0: mAP50 83.993%, mAP50-95 59.683%, AP75 69.845%, AP_small 32.795%.

## License

Checkpoints and code are **AGPL-3.0**. Keep the root `LICENSE` and `NOTICE` on
redistribution.