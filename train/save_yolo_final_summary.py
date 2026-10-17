import argparse
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import yaml


def _label_path_for_image(img_path: Path) -> Path:
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return Path(str(img_path).replace("images", "labels")).with_suffix(".txt")


def _count_images_instances(val_images_dir: Path, class_num: int):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = [p for p in val_images_dir.rglob("*") if p.suffix.lower() in exts]
    class_images = defaultdict(int)
    class_instances = defaultdict(int)

    for img_path in image_paths:
        label_path = _label_path_for_image(img_path)
        if not label_path.exists():
            continue
        appeared = set()
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(float(parts[0]))
                if 0 <= cls_id < class_num:
                    appeared.add(cls_id)
                    class_instances[cls_id] += 1
        for cls_id in appeared:
            class_images[cls_id] += 1

    return len(image_paths), sum(class_instances.values()), class_images, class_instances


def _get_best_epoch_info(results_csv: Path):
    if not results_csv.exists():
        return None

    df = pd.read_csv(results_csv)
    map_col = "metrics/mAP50-95(B)"
    if map_col not in df.columns or df.empty:
        return None

    best_idx = df[map_col].idxmax()
    best_row = df.loc[best_idx]
    raw_epoch = int(best_row["epoch"])
    best_epoch = raw_epoch + 1 if raw_epoch == best_idx else raw_epoch
    completed_hours = float(df.iloc[-1].get("time", 0.0)) / 3600.0

    return {
        "best_epoch": best_epoch,
        "completed_epochs": len(df),
        "best_map5095": float(best_row[map_col]),
        "completed_hours": completed_hours,
    }


def _resolve_val_images_dir(data_yaml: str) -> Path:
    with open(data_yaml, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    root = Path(data_cfg.get("path", Path(data_yaml).parent))
    val_rel = Path(data_cfg.get("val", "val"))
    return val_rel if val_rel.is_absolute() else root / val_rel


def _size_mb(path: Path) -> str:
    return f"{os.path.getsize(path) / 1024 / 1024:.1f}MB" if path.exists() else "N/A"


def _environment_line(device: str) -> str:
    import ultralytics

    cuda_str = "CPU"
    if torch.cuda.is_available():
        try:
            gpu_index = int(device)
        except ValueError:
            gpu_index = 0
        gpu = torch.cuda.get_device_name(gpu_index)
        mem = torch.cuda.get_device_properties(gpu_index).total_memory / 1024**2
        cuda_str = f"CUDA:{device} ({gpu}, {int(mem)}MiB)"

    return (
        f"Ultralytics {ultralytics.__version__}  "
        f"Python-{platform.python_version()} "
        f"torch-{torch.__version__} "
        f"{cuda_str}"
    )


def _model_summary(model, imgsz: int) -> str:
    from ultralytics.utils.torch_utils import get_flops, get_num_gradients, get_num_params

    nn_model = getattr(model, "model", model) if not hasattr(model, "yaml") else model
    n_p = get_num_params(nn_model)
    n_g = get_num_gradients(nn_model)
    n_l = sum(1 for _, m in nn_model.named_modules() if len(m._modules) == 0)
    flops = get_flops(nn_model, imgsz)
    fused = " (fused)" if getattr(nn_model, "is_fused", lambda: False)() else ""
    fs = f", {flops:.1f} GFLOPs" if flops else ""
    yaml_file = getattr(nn_model, "yaml_file", "") or getattr(nn_model, "yaml", {}).get("yaml_file", "")
    model_name = Path(yaml_file).stem.replace("yolo", "YOLO") if yaml_file else "Model"
    return f"{model_name} summary{fused}: {n_l:,} layers, {n_p:,} parameters, {n_g:,} gradients{fs}"


def _load_fused_checkpoint_model(checkpoint: Path):
    """Load best.pt as a fused inference model for summary text only."""
    try:
        from ultralytics.nn.tasks import load_checkpoint

        model, _ = load_checkpoint(str(checkpoint), device="cpu", fuse=False)
        if hasattr(model, "fuse"):
            model = model.fuse(verbose=False)
        return model
    except Exception:
        return None


def _names_as_items(names):
    if isinstance(names, dict):
        return [(int(k), v) for k, v in names.items()]
    return list(enumerate(names))


def _class_metric(metric_values, class_index, default=0.0):
    try:
        return float(metric_values[class_index])
    except (IndexError, KeyError, TypeError):
        return default


def _class_metric_maps(metrics):
    box = metrics.box
    class_ids = getattr(box, "ap_class_index", None)
    if class_ids is None:
        return None

    class_ids = [int(c) for c in class_ids]
    return {
        int(cls_id): {
            "p": _class_metric(box.p, i),
            "r": _class_metric(box.r, i),
            "ap50": _class_metric(box.ap50, i),
            "ap": _class_metric(box.ap, i),
        }
        for i, cls_id in enumerate(class_ids)
    }


def write_summary(
    exp_dir,
    data_yaml,
    imgsz=640,
    batch=12,
    device="0",
    patience=100,
    total_epochs=300,
    yolo_model=None,
    metrics=None,
):
    exp_dir = Path(exp_dir)
    best_pt = exp_dir / "weights" / "best.pt"
    last_pt = exp_dir / "weights" / "last.pt"
    results_csv = exp_dir / "results.csv"
    summary_txt = exp_dir / "final_summary.txt"

    if not best_pt.exists():
        print(f"  [SKIP] best.pt not found: {best_pt}")
        return None

    model = yolo_model
    summary_model = _load_fused_checkpoint_model(best_pt) or model
    has_metrics = metrics is not None and hasattr(metrics, "box")

    best_info = _get_best_epoch_info(results_csv)
    names_src = getattr(summary_model, "names", None) or (getattr(model, "names", None) if model is not None else {})
    names = dict(_names_as_items(names_src)) if names_src else {}
    if has_metrics and names:
        total_images, total_instances, cls_images, cls_instances = _count_images_instances(
            _resolve_val_images_dir(data_yaml), len(names)
        )
        class_metrics = _class_metric_maps(metrics)
        speed = metrics.speed if hasattr(metrics, "speed") else {}
    else:
        total_images, total_instances, cls_images, cls_instances = 0, 0, defaultdict(int), defaultdict(int)
        class_metrics, speed = None, {}

    # Use a BOM for Windows Notepad/Excel-compatible UTF-8 text display.
    with open(summary_txt, "w", encoding="utf-8-sig") as f:
        if best_info and best_info["completed_epochs"] < total_epochs:
            f.write(
                "EarlyStopping: Training stopped early as no improvement observed "
                f"in last {patience} epochs. Best results observed at epoch "
                f"{best_info['best_epoch']}, best model saved as best.pt.\n"
            )
            f.write(
                f"To update EarlyStopping(patience={patience}) pass a new patience "
                "value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.\n\n"
            )

        if best_info:
            f.write(f"{best_info['completed_epochs']} epochs completed in {best_info['completed_hours']:.3f} hours.\n")
        else:
            f.write("? epochs completed.\n")
        f.write(f"Optimizer stripped from {last_pt}, {_size_mb(last_pt)}\n")
        f.write(f"Optimizer stripped from {best_pt}, {_size_mb(best_pt)}\n\n")
        f.write(f"Best checkpoint: {best_pt}\n")

        if has_metrics and summary_model is not None:
            f.write(_environment_line(device) + "\n")
            f.write(_model_summary(summary_model, imgsz) + "\n")
            f.write(f"{'Class':>20}{'Images':>10}{'Instances':>12}{'Box(P':>12}{'R':>10}{'mAP50':>10}{'mAP50-95':>12}\n")

            box = metrics.box
            f.write(
                f"{'all':>20}{total_images:>10}{total_instances:>12}"
                f"{float(box.mp):>12.3g}{float(box.mr):>10.3g}{float(box.map50):>10.3g}{float(box.map):>12.3g}\n"
            )

            for cls_id, name in names.items():
                if class_metrics is not None:
                    values = class_metrics.get(cls_id, {"p": 0.0, "r": 0.0, "ap50": 0.0, "ap": 0.0})
                else:
                    values = {
                        "p": _class_metric(box.p, cls_id),
                        "r": _class_metric(box.r, cls_id),
                        "ap50": _class_metric(box.ap50, cls_id),
                        "ap": _class_metric(box.ap, cls_id),
                    }
                f.write(
                    f"{name:>20}{cls_images[cls_id]:>10}{cls_instances[cls_id]:>12}"
                    f"{values['p']:>12.3g}{values['r']:>10.3g}{values['ap50']:>10.3g}{values['ap']:>12.3g}\n"
                )

            f.write("\n")
            f.write(
                f"Speed: {speed.get('preprocess', 0):.1f}ms preprocess, "
                f"{speed.get('inference', 0):.1f}ms inference, "
                f"{speed.get('loss', 0):.1f}ms loss, "
                f"{speed.get('postprocess', 0):.1f}ms postprocess per image\n"
            )
        f.write(f"Results saved to {exp_dir}\n")

    print(f"  Final summary saved to: {summary_txt}")
    return str(summary_txt)


def main():
    parser = argparse.ArgumentParser(description="Generate final_summary.txt for a YOLO experiment")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--data_yaml", default=None)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--total_epochs", type=int, default=300)
    args = parser.parse_args()

    root = r"<original-workspace>"
    sys.path.insert(0, root)
    data_yaml = args.data_yaml or os.path.join(root, "mytomato.yaml")
    write_summary(
        exp_dir=args.exp_dir,
        data_yaml=data_yaml,
        patience=args.patience,
        total_epochs=args.total_epochs,
    )


if __name__ == "__main__":
    main()
