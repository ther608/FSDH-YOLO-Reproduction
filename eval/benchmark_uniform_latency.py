"""Benchmark existing Ultralytics checkpoints with one controlled protocol."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def preprocess(image_path: Path, imgsz: int) -> torch.Tensor:
    from ultralytics.data.augment import LetterBox

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read benchmark image: {image_path}")
    image = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=image)
    array = np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0).float().div_(255.0)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def benchmark(weights: Path, tensor: torch.Tensor, warmup: int, iterations: int) -> dict:
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.torch_utils import get_flops

    if not torch.cuda.is_available():
        raise RuntimeError("Uniform latency benchmark requires CUDA")
    device = torch.device("cuda:0")
    model = YOLO(str(weights)).model.to(device).float().eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    model = model.fuse(verbose=False).eval()
    fused_parameters = sum(parameter.numel() for parameter in model.parameters())
    tensor = tensor.to(device)
    nc = int(getattr(model, "nc", 3))
    torch.backends.cudnn.benchmark = True

    with torch.inference_mode():
        for _ in range(warmup):
            output = model(tensor)
            inference = output[0] if isinstance(output, tuple) else output
            non_max_suppression(inference, conf_thres=0.25, iou_thres=0.7, max_det=300, nc=nc)
        torch.cuda.synchronize()

        forward_ms: list[float] = []
        nms_ms: list[float] = []
        total_ms: list[float] = []
        detections: list[int] = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            output = model(tensor)
            torch.cuda.synchronize()
            after_forward = time.perf_counter_ns()
            inference = output[0] if isinstance(output, tuple) else output
            result = non_max_suppression(
                inference, conf_thres=0.25, iou_thres=0.7, max_det=300, nc=nc
            )
            torch.cuda.synchronize()
            end = time.perf_counter_ns()
            fwd = (after_forward - start) / 1e6
            nms = (end - after_forward) / 1e6
            forward_ms.append(fwd)
            nms_ms.append(nms)
            total_ms.append(fwd + nms)
            detections.append(len(result[0]))

    gflops = float(get_flops(model, imgsz=tensor.shape[-1]))
    forward_mean = statistics.mean(forward_ms)
    total_mean = statistics.mean(total_ms)
    return {
        "params_M": parameters / 1e6,
        "params_fused_M": fused_parameters / 1e6,
        "gflops": gflops,
        "forward_ms_mean": forward_mean,
        "forward_ms_sd": statistics.stdev(forward_ms),
        "forward_ms_median": statistics.median(forward_ms),
        "forward_ms_p95": percentile(forward_ms, 95),
        "nms_ms_mean": statistics.mean(nms_ms),
        "nms_ms_sd": statistics.stdev(nms_ms),
        "total_ms_mean": total_mean,
        "total_ms_sd": statistics.stdev(total_ms),
        "total_ms_median": statistics.median(total_ms),
        "total_ms_p95": percentile(total_ms, 95),
        "fps_from_forward_mean": 1000.0 / forward_mean,
        "fps_from_total_mean": 1000.0 / total_mean,
        "detections_mean": statistics.mean(detections),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest = list(csv.DictReader(handle))
    if not manifest:
        raise RuntimeError(f"Manifest is empty: {args.manifest}")

    print("=== Uniform CUDA latency benchmark ===")
    print(f"Device      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable'}")
    print("Precision   : FP32")
    print("Batch size  : 1")
    print(f"Input size  : {args.imgsz}x{args.imgsz} (unless overridden by a manifest row)")
    print(f"Warmup      : {args.warmup}")
    print(f"Iterations  : {args.iterations}")
    print("Tensor      : constructed and moved to CUDA before warmup/timing")
    print("Timing      : CUDA synchronize before and after every measured forward")
    print("Primary FPS : 1000 / mean forward latency (model forward including decode)")
    print("Auxiliary   : forward + class-aware NMS is also reported")
    print()

    tensors: dict[int, torch.Tensor] = {}
    rows = []
    for item in manifest:
        item_imgsz = int(item.get("imgsz") or args.imgsz)
        tensor = tensors.setdefault(item_imgsz, preprocess(args.image, item_imgsz))
        weights = Path(item["weights"])
        if not weights.is_absolute():
            weights = REPO_ROOT / weights
        result = benchmark(weights, tensor, args.warmup, args.iterations)
        row = {
            "section": item["section"],
            "model": item["model"],
            "seed": int(item["seed"]) if item.get("seed", "") != "" else "",
            "weights": str(weights),
            "imgsz": item_imgsz,
            **result,
        }
        rows.append(row)
        print(
            f"{item['model']} [seed={row['seed']}]: "
            f"FPS_forward={result['fps_from_forward_mean']:.2f}, "
            f"forward={result['forward_ms_mean']:.3f}±{result['forward_ms_sd']:.3f} ms; "
            f"FPS_forward+NMS={result['fps_from_total_mean']:.2f}, "
            f"forward+NMS={result['total_ms_mean']:.3f}±{result['total_ms_sd']:.3f} ms"
        )
        torch.cuda.empty_cache()

    fields = list(rows[0])
    with (args.out / "uniform_latency_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "protocol": {
            "device": torch.cuda.get_device_name(0),
            "precision": "FP32",
            "batch": 1,
            "input": "per-manifest; see results.imgsz",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "input_image": str(args.image),
            "model_fused": True,
            "input_tensor_preconstructed": True,
            "input_tensor_on_cuda_before_timing": True,
            "cuda_synchronization": "before each timed forward, after forward, and after NMS",
            "parameter_count_scope": "unfused trainable architecture; params_fused_M is retained separately",
            "forward_scope": "model forward including decode",
            "total_scope": "model forward plus class-aware NMS",
            "preprocessing_included": False,
            "nms_confidence": 0.25,
            "nms_iou": 0.7,
            "max_det": 300,
        },
        "results": rows,
    }
    (args.out / "uniform_latency_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)
    aggregate_rows = []
    metric_fields = [
        "forward_ms_mean",
        "nms_ms_mean",
        "total_ms_mean",
        "total_ms_p95",
        "fps_from_forward_mean",
        "fps_from_total_mean",
    ]
    for model_name, model_rows in sorted(grouped.items()):
        output = {
            "model": model_name,
            "n": len(model_rows),
            "seeds": ",".join(str(row["seed"]) for row in sorted(model_rows, key=lambda value: value["seed"])),
            "params_M": statistics.mean(row["params_M"] for row in model_rows),
            "gflops": statistics.mean(row["gflops"] for row in model_rows),
        }
        for field in metric_fields:
            values = [row[field] for row in model_rows]
            output[f"{field}_mean"] = statistics.mean(values)
            output[f"{field}_sample_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(output)
    if aggregate_rows:
        with (args.out / "uniform_latency_three_seed_mean_sd.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
            writer.writeheader()
            writer.writerows(aggregate_rows)
        (args.out / "uniform_latency_three_seed_mean_sd.json").write_text(
            json.dumps(
                {
                    "std_definition": "sample standard deviation, ddof=1",
                    "results": aggregate_rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("\n=== Three-seed FPS summary (mean ± sample SD) ===")
        print(
            f"{'Model':<22} {'Params/M':>10} {'FPS forward':>22} "
            f"{'FPS forward+NMS':>22}"
        )
        print("-" * 80)
        for row in aggregate_rows:
            print(
                f"{row['model']:<22} {row['params_M']:>10.3f} "
                f"{row['fps_from_forward_mean_mean']:>9.2f} ± "
                f"{row['fps_from_forward_mean_sample_sd']:<8.2f} "
                f"{row['fps_from_total_mean_mean']:>9.2f} ± "
                f"{row['fps_from_total_mean_sample_sd']:<8.2f}"
            )
        print(f"\nDetailed results saved to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
