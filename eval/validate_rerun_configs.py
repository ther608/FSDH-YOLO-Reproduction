"""Validate all custom model configurations before paper-scale reruns."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


MODEL_ROOT = ROOT / "ultralytics" / "cfg" / "models" / "11"
CONFIGS = [
    "yolo11n_capacity_matched.yaml",
    "yolo11n-CA-P3.yaml",
    "yolo11n-CA-P4.yaml",
    "yolo11n-CA.yaml",
    "yolo11n-CA-P3P4P5.yaml",
    "yolo11n_SHAB-local.yaml",
    "yolo11n_SHAB-local-mid.yaml",
    "yolo11n_SHAB.yaml",
    "yolo11n_SPDConv.yaml",
    "yolo11n_SPDConv-SHAB.yaml",
    "yolo11n_SPDConv-SHAB-mid.yaml",
    "yolo11n_MaturityDetect.yaml",
    "yolo11n_CA-MaturityDetect.yaml",
    "yolo11n_MaturityDetect-SHAB.yaml",
    "yolo11n_MaturityDetect-SHAB-lambda005.yaml",
    "yolo11n_CA-MaturityDetect-SHAB-lambda0.yaml",
    "yolo11n_CA-MaturityDetect-SHAB-lambda005.yaml",
    "yolo11n_CA-MaturityDetect-SHAB.yaml",
    "yolo11n_CA-MaturityDetect-SHAB-lambda020.yaml",
    "yolo11s_CA-MaturityDetect-SHAB.yaml",
    "yolo11n_WTConv-MaturityDetect-SHAB-lambda005.yaml",
    "yolo11n_FasterBlock-MaturityDetect-SHAB-lambda005.yaml",
    "yolo11n_DySample-MaturityDetect-SHAB-lambda005.yaml",
]


def numerical_probe() -> dict[str, float | bool]:
    """Reproduce the old FP16 overflow and verify the FP32 bounded formulation."""
    mu_half = torch.tensor([2.0], dtype=torch.float16)
    target_half = torch.tensor([0.0], dtype=torch.float16)
    raw_lv_half = torch.tensor([-13.25], dtype=torch.float16)
    old_var = F.softplus(raw_lv_half) + 1e-6
    old_loss = 0.5 * ((mu_half - target_half).square() / old_var + old_var.log())

    mu = mu_half.float()
    target = target_half.float()
    variance = F.softplus(raw_lv_half.float()).clamp(1e-4, 1e2)
    new_loss = 0.5 * ((mu - target).square() / variance + variance.log())
    return {
        "old_fp16_loss_finite": bool(torch.isfinite(old_loss).all()),
        "new_fp32_loss_finite": bool(torch.isfinite(new_loss).all()),
        "new_variance": float(variance.item()),
        "new_loss": float(new_loss.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = []
    for name in CONFIGS:
        path = MODEL_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        yolo = YOLO(str(path))
        model = yolo.model
        parameters = sum(parameter.numel() for parameter in model.parameters())
        gflops = float(get_flops(model, imgsz=640))
        modules = list(model.modules())
        coord_modules = [module for module in modules if module.__class__.__name__ == "CoordAtt"]
        shab_modules = [module for module in modules if module.__class__.__name__ == "SHAB"]
        maturity_heads = [module for module in modules if module.__class__.__name__ == "MaturityDetect"]
        wtconv_modules = [module for module in modules if module.__class__.__name__ == "WTConv2d"]
        faster_modules = [module for module in modules if module.__class__.__name__ == "FasterBlock"]
        dysample_modules = [module for module in modules if module.__class__.__name__ == "DySample"]
        spdconv_modules = [module for module in modules if module.__class__.__name__ == "SPDConv"]
        if "CA" in name:
            if not coord_modules:
                raise AssertionError(f"{name}: expected CoordAtt modules")
            if any(not isinstance(getattr(module, "act", None), torch.nn.ReLU6) for module in coord_modules):
                raise AssertionError(f"{name}: every CoordAtt must use ReLU6")
        if "SHAB" in name:
            if "local-mid" in name:
                expected_branches = 2
            elif "SHAB-mid" in name or "local" in name:
                expected_branches = 1
            else:
                expected_branches = 3
            if not shab_modules or any(module.branches != expected_branches for module in shab_modules):
                raise AssertionError(f"{name}: unexpected SHAB branch configuration")
        if "MaturityDetect" in name:
            if len(maturity_heads) != 1:
                raise AssertionError(f"{name}: expected exactly one MaturityDetect head")
            yaml = model.yaml
            for key in ("maturity_targets", "maturity_lambda", "maturity_variance_min", "maturity_variance_max"):
                if key not in yaml:
                    raise AssertionError(f"{name}: missing explicit {key}")
        if "WTConv" in name and len(wtconv_modules) != 2:
            raise AssertionError(f"{name}: expected two WTConv2d modules at P3/P4")
        if "FasterBlock" in name and len(faster_modules) != 2:
            raise AssertionError(f"{name}: expected two FasterBlock modules at P3/P4")
        if "DySample" in name and len(dysample_modules) != 2:
            raise AssertionError(f"{name}: expected two DySample top-down upsamplers")
        if "SPDConv" in name and len(spdconv_modules) != 1:
            raise AssertionError(f"{name}: expected exactly one SPDConv module")
        rows.append(
            {
                "config": name,
                "parameters": parameters,
                "params_m": parameters / 1e6,
                "gflops": gflops,
                "coordatt_modules": len(coord_modules),
                "shab_modules": len(shab_modules),
                "maturitydetect_heads": len(maturity_heads),
                "wtconv_modules": len(wtconv_modules),
                "fasterblock_modules": len(faster_modules),
                "dysample_modules": len(dysample_modules),
                "spdconv_modules": len(spdconv_modules),
            }
        )

    probe = numerical_probe()
    if probe["old_fp16_loss_finite"]:
        raise AssertionError("The diagnostic input no longer reproduces the legacy FP16 overflow.")
    if not probe["new_fp32_loss_finite"]:
        raise AssertionError("The bounded FP32 Gaussian NLL is non-finite.")
    by_name = {row["config"]: row for row in rows}
    matched = by_name["yolo11n_capacity_matched.yaml"]["parameters"]
    full = by_name["yolo11n_CA-MaturityDetect-SHAB.yaml"]["parameters"]
    capacity_relative_difference = abs(matched - full) / full
    if capacity_relative_difference > 0.05:
        raise AssertionError(
            f"Capacity baseline differs from MSCA-YOLO-n by {capacity_relative_difference:.2%}, exceeding 5%."
        )
    report = {
        "configs": rows,
        "capacity_match": {
            "baseline_parameters": matched,
            "full_parameters": full,
            "relative_difference": capacity_relative_difference,
            "tolerance": 0.05,
        },
        "gaussian_nll_probe": probe,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
