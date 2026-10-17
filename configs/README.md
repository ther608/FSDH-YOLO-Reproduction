# Model configs

**Naming aliases.** In the paper, the input module is **FSDConv** and the
hierarchy block is **SHEB**. In code and released checkpoints they are
**`SPOConv`** and **`SHAB`**, respectively. The aliases are intentional and are
kept for checkpoint compatibility; they are not different modules.

These YAMLs are a paper-related subset of
`src/ultralytics/cfg/models/11/`. Training works with either this directory or
the source package; contents are identical.

## Main model

- `yolo11n_FSDConv-SHAB.yaml`: **FSDH-YOLO**
  - layer 1 `SPOConv` (FSDConv) replaces the input stem;
  - `SHAB` (SHEB) is inserted after backbone stages P3/P4/P5, before the Neck;
  - the rest is standard YOLO11n.

## Ablation

- `yolo11n_FSDConv.yaml`: FSDConv only
- `yolo11n_SHAB.yaml`: full SHEB (L+M+G)
- `yolo11n_SHAB-local.yaml` / `-mid` / `-global`: SHEB-L / M / G
- `yolo11n_SHAB-local-mid.yaml` / `-local-global` / `-mid-global`: SHEB-LM / LG / MG
- `yolo11n_FSDConv-SHAB-mid.yaml`: FSDConv + SHEB-M
- `yolo11n_PM.yaml`: capacity-matched YOLO11n-PM (width 0.25->0.27, max 1024->992)
- `yolo11n_capacity_matched.yaml`: alternative capacity control

## SPDConv controls

- `yolo11n_SPDConv.yaml`: SPDConv only
- `yolo11n_SPDConv-SHAB.yaml`: SPDConv + full SHEB
- `yolo11n_SPDConv-SHAB-mid.yaml`: SPDConv + SHEB-M

## Multi-scale / feature-enhancement controls

- `yolo11n_ASPP.yaml`
- `yolo11n_SPP-Compare.yaml`
- `yolo11n_DilatedConv.yaml`
- `yolo11n_RepConv.yaml`
- `yolo11n_MultiScaleDWConv.yaml`
- `yolo11n_RFB.yaml`

## Baseline

- `yolo11.yaml`: YOLO11 architecture (n scale via `yolo11n.pt` or `scale: n`)

## Notes

- The main config uses `nc: 3` (Dataset A). Dataset B has 4 classes; the training
  data YAML overrides the head class count.
- Paper `SHEB` is code `SHAB`; paper `FSDConv` is code `SPOConv`. These are naming
  compatibility aliases, not different modules.