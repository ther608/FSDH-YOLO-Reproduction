import argparse
import gc
import json
import os
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, os.path.join(ROOT, "src"))

from save_yolo_final_summary import write_summary
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import init_seeds, intersect_dicts


DEFAULT_DATA_YAML = (
    r"<data-root>/data\An RGB-D Image Dataset for Lychee Detection and Ma"
    r"\An RGB-D Image Dataset for Lychee Detection and Ma"
    r"\An RGB-D Image Dataset for Lychee\lychee.yaml"
)
PROJECT_DIR = os.path.join(ROOT, "runs", "detect", "lychee")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(ROOT, "yolo11m.pt"))
    parser.add_argument("--name", default="baseline_11m")
    parser.add_argument("--weights", default=None, help="Pretrained weights to load, e.g. yolo11s.pt")
    parser.add_argument(
        "--weights-map",
        default="none",
        choices=[
            "none",
            "peanet_yolov5d_repro",
            "fpdw_yolov8_repro",
            "edge_yolov11_repro",
            "gpc_yolov8_repro",
            "fsdh_yolo11n",
        ],
        help="Optional layer-index remap when paper-reproduction modules insert layers into a pretrained base model",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--val-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="Training random seed")
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the training split to use (1.0 for all data; smaller values are for smoke tests)",
    )

    parser.add_argument("--use-focal", action="store_true", default=False)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--use-varifocal", action="store_true", default=False)
    parser.add_argument("--use-qfl", action="store_true", default=False)

    parser.add_argument(
        "--iou-type", default="ciou", choices=["ciou", "diou", "eiou", "siou", "mpdiou", "inner_iou", "piou"]
    )
    parser.add_argument("--inner-iou-ratio", type=float, default=0.7)

    parser.add_argument("--use-wiou", action="store_true", default=False)
    parser.add_argument("--use-focal-wiou", action="store_true", default=False)

    parser.add_argument("--use-nwd", action="store_true", default=False)
    parser.add_argument("--nwd-ratio", type=float, default=0.5)
    parser.add_argument("--nwd-constant", type=float, default=12.8)
    parser.add_argument(
        "--use-gcd",
        action="store_true",
        default=False,
        help="Replace the CIoU box-regression term with Gaussian Combined Distance; DFL remains enabled",
    )

    parser.add_argument("--data", default=DEFAULT_DATA_YAML, help="Path to dataset YAML")
    parser.add_argument(
        "--project",
        default=PROJECT_DIR,
        help="Root directory for training results (default: legacy lychee result directory)",
    )
    parser.add_argument("--ordinal", action="store_true", default=False)
    parser.add_argument(
        "--enable-ordinal-loss",
        "--use-oql",
        dest="use_oql",
        action="store_true",
        default=False,
        help=(
            "Enable the auxiliary expected-stage SmoothL1 ordinal loss for ordered ripeness classes. "
            "Disabled by default; --use-oql is retained as a backward-compatible alias."
        ),
    )
    parser.add_argument("--oql-lambda", type=float, default=0.2,
                        help="Expected-stage ordinal-loss weight (added to, not replacing, YOLO cls loss)")
    parser.add_argument("--oql-expect-weight", type=float, default=1.0,
                        help="Weight for the expected-stage regression term")
    parser.add_argument("--oql-kl-weight", type=float, default=0.0,
                        help="Weight for the ordinal KL-divergence term (0 = disabled)")
    parser.add_argument("--oql-neighbor", type=float, default=0.1,
                        help="Neighbour-class smoothing for ordinal distribution")
    parser.add_argument("--oql-tau", type=float, default=0.5,
                        help="Temperature for ordinal distance decay in KL term")
    parser.add_argument("--use-qcl", action="store_true", default=False,
                        help="Enable Quality Consistency Loss")
    parser.add_argument("--qcl-lambda", type=float, default=0.1,
                        help="QCL loss weight")
    parser.add_argument("--qcl-type", default="bce", choices=["bce", "l1", "smoothl1"],
                        help="QCL loss type")
    parser.add_argument("--qcl-detach-iou", action="store_true", default=True,
                        help="Detach IoU target for QCL")
    parser.add_argument("--no-qcl-detach-iou", action="store_false", dest="qcl_detach_iou",
                        help="Do not detach IoU target")
    parser.add_argument("--qcl-min-iou", type=float, default=0.0,
                        help="Minimum IoU clamp for QCL target")
    parser.add_argument("--tal-alpha", type=float, default=0.5,
                        help="TaskAlignedAssigner classification exponent. TOOD-style default is often 1.0")
    parser.add_argument("--tal-beta", type=float, default=6.0,
                        help="TaskAlignedAssigner IoU exponent")
    parser.add_argument("--tal-topk", type=int, default=10,
                        help="TaskAlignedAssigner top-k positive candidates")
    parser.add_argument("--tal-topk2", type=int, default=None,
                        help="Optional secondary top-k filter for TaskAlignedAssigner")
    parser.add_argument("--soft-nms", action="store_true", default=False)
    parser.add_argument("--soft-nms-method", default="linear", choices=["linear", "gaussian"])
    parser.add_argument("--soft-nms-sigma", type=float, default=0.5)
    parser.add_argument("--pf-prompt-lambda", type=float, default=0.1,
                        help="Auxiliary class-agnostic fruitness loss for PF-CSP")
    parser.add_argument("--pf-quality-lambda", type=float, default=0.2,
                        help="IoU-quality branch loss weight for PF-CGQS")
    parser.add_argument("--pf-centerness-lambda", type=float, default=0.2,
                        help="Centerness branch loss weight for PF-CGQS")
    parser.add_argument("--pf-roi-lambda", type=float, default=1.0,
                        help="ROI maturity-classification loss weight for Where-What")
    parser.add_argument("--pf-proposal-topk", type=int, default=300,
                        help="Number of class-agnostic fruit proposals classified at inference")
    parser.add_argument("--use-dgad", action="store_true", default=False,
                        help="Enable the training-only P3 monocular-depth auxiliary head")
    parser.add_argument("--dgad-source-root", default=None,
                        help="Dataset-A source root containing Image_YoloLabel_*/depth/gray")
    parser.add_argument("--dgad-augmentation-log", default=None,
                        help="CSV used to replay the frozen offline geometric transforms on depth")
    parser.add_argument("--dgad-lambda", type=float, default=0.02)
    parser.add_argument("--dgad-gradient-weight", type=float, default=0.1)
    parser.add_argument("--dgad-feature-layer", type=int, default=5)
    parser.add_argument("--dgad-feature-channels", type=int, default=128)
    return parser.parse_args()


def configure_loss_environment(args):
    if args.use_gcd and (args.use_nwd or args.use_wiou or args.use_focal_wiou):
        raise ValueError("--use-gcd cannot be combined with --use-nwd/--use-wiou/--use-focal-wiou")

    os.environ["YOLO_VAL_WORKERS"] = str(args.val_workers)
    os.environ["YOLO_USE_GCD"] = "1" if args.use_gcd else "0"
    os.environ["YOLO_USE_NWD"] = "1" if args.use_nwd else "0"
    os.environ["YOLO_NWD_RATIO"] = str(args.nwd_ratio)
    os.environ["YOLO_NWD_CONSTANT"] = str(args.nwd_constant)
    os.environ["YOLO_USE_WIOU"] = "1" if args.use_wiou else "0"
    os.environ["YOLO_USE_FOCAL_WIOU"] = "1" if args.use_focal_wiou else "0"
    os.environ["YOLO_IOU_TYPE"] = args.iou_type
    os.environ["YOLO_INNER_IOU_RATIO"] = str(args.inner_iou_ratio)
    os.environ["YOLO_ORDINAL"] = "1" if args.ordinal else "0"
    # Expected-stage ordinal loss (legacy internal/environment name: OQL).
    os.environ["YOLO_USE_OQL"] = "1" if args.use_oql else "0"
    os.environ["YOLO_OQL_LAMBDA"] = str(args.oql_lambda)
    os.environ["YOLO_OQL_EXPECT_WEIGHT"] = str(args.oql_expect_weight)
    os.environ["YOLO_OQL_KL_WEIGHT"] = str(args.oql_kl_weight)
    os.environ["YOLO_OQL_NEIGHBOR"] = str(args.oql_neighbor)
    os.environ["YOLO_OQL_TAU"] = str(args.oql_tau)
    # QCL — Quality Consistency Loss
    os.environ["YOLO_USE_QCL"] = "1" if args.use_qcl else "0"
    os.environ["YOLO_QCL_LAMBDA"] = str(args.qcl_lambda)
    os.environ["YOLO_QCL_TYPE"] = args.qcl_type
    os.environ["YOLO_QCL_DETACH_IOU"] = "1" if args.qcl_detach_iou else "0"
    os.environ["YOLO_QCL_MIN_IOU"] = str(args.qcl_min_iou)
    os.environ["YOLO_TAL_ALPHA"] = str(args.tal_alpha)
    os.environ["YOLO_TAL_BETA"] = str(args.tal_beta)
    os.environ["YOLO_TAL_TOPK"] = str(args.tal_topk)
    os.environ["YOLO_TAL_TOPK2"] = "" if args.tal_topk2 is None else str(args.tal_topk2)
    # Keep the requested formulation explicit: hard YOLO classification targets plus
    # expected-stage SmoothL1, rather than the legacy ordinal soft-label targets.
    if args.use_oql and args.ordinal:
        print(
            "[WARNING] --enable-ordinal-loss and --ordinal both set — expected-stage SmoothL1 takes priority, "
            "disabling ordinal soft-labels"
        )
        os.environ["YOLO_ORDINAL"] = "0"
    os.environ["YOLO_USE_VARIFOCAL"] = "1" if args.use_varifocal else "0"
    os.environ["YOLO_USE_QFL"] = "1" if args.use_qfl else "0"
    os.environ["YOLO_SOFT_NMS"] = "1" if args.soft_nms else "0"
    os.environ["YOLO_SOFT_NMS_METHOD"] = args.soft_nms_method
    os.environ["YOLO_SOFT_NMS_SIGMA"] = str(args.soft_nms_sigma)
    os.environ["YOLO_PF_PROMPT_LAMBDA"] = str(args.pf_prompt_lambda)
    os.environ["YOLO_QHEAD_LAMBDA"] = str(args.pf_quality_lambda)
    os.environ["YOLO_PF_CENTERNESS_LAMBDA"] = str(args.pf_centerness_lambda)
    os.environ["YOLO_PF_ROI_LAMBDA"] = str(args.pf_roi_lambda)
    os.environ["YOLO_PF_PROPOSAL_TOPK"] = str(args.pf_proposal_topk)
    if args.use_dgad and (not args.dgad_source_root or not args.dgad_augmentation_log):
        raise ValueError("--use-dgad requires --dgad-source-root and --dgad-augmentation-log")
    os.environ["YOLO_USE_DGAD"] = "1" if args.use_dgad else "0"
    os.environ["YOLO_DGAD_SOURCE_ROOT"] = args.dgad_source_root or ""
    os.environ["YOLO_DGAD_AUG_LOG"] = args.dgad_augmentation_log or ""
    os.environ["YOLO_DGAD_LAMBDA"] = str(args.dgad_lambda)
    os.environ["YOLO_DGAD_GRADIENT_WEIGHT"] = str(args.dgad_gradient_weight)
    os.environ["YOLO_DGAD_FEATURE_LAYER"] = str(args.dgad_feature_layer)
    os.environ["YOLO_DGAD_FEATURE_CHANNELS"] = str(args.dgad_feature_channels)


def print_run_banner(args):
    print(f"[INFO] Model:       {args.model}")
    print(f"[INFO] Name:        {args.name}")
    print(f"[INFO] Weights:     {args.weights}")
    print(f"[INFO] Weights map: {args.weights_map}")
    print(f"[INFO] Data:        {args.data}")
    print(f"[INFO] Epochs:      {args.epochs}")
    print(f"[INFO] Batch:       {args.batch}")
    print(f"[INFO] Image size:  {args.imgsz}")
    print(f"[INFO] Workers:     {args.workers}")
    print(f"[INFO] Val workers: {args.val_workers}")
    print(f"[INFO] Patience:    {args.patience}")
    print(f"[INFO] Seed:        {args.seed}")
    print(f"[INFO] Focal Loss:  {args.use_focal}")
    print(f"[INFO] IoU type:    {args.iou_type}")
    print(f"[INFO] WIoU:        {args.use_wiou}")
    print(f"[INFO] Focal-WIoU:  {args.use_focal_wiou}")
    print(f"[INFO] NWD:         {args.use_nwd}")
    print(f"[INFO] GCD:         {args.use_gcd}")
    print(f"[INFO] Ordinal:     {args.ordinal}")
    print(f"[INFO] Ordinal loss:{args.use_oql} (expected-stage SmoothL1)")
    print(f"[INFO] QCL:         {args.use_qcl}")
    print(f"[INFO] DGAD:        {args.use_dgad} (lambda={args.dgad_lambda}, P3 layer={args.dgad_feature_layer})")
    print(f"[INFO] TAL:         alpha={args.tal_alpha}, beta={args.tal_beta}, "
          f"topk={args.tal_topk}, topk2={args.tal_topk2}")
    print(
        "[INFO] PF adapters: "
        f"prompt={args.pf_prompt_lambda}, quality={args.pf_quality_lambda}, "
        f"centerness={args.pf_centerness_lambda}, roi={args.pf_roi_lambda}, "
        f"proposal_topk={args.pf_proposal_topk}"
    )

    if args.use_focal:
        print(f"[LOSS] Focal Loss (gamma={args.focal_gamma}, alpha={args.focal_alpha})")
    if args.use_varifocal:
        print("[LOSS] Varifocal Loss (gamma=2.0, alpha=0.75)")
    if args.use_qfl:
        print("[LOSS] Quality Focal Loss (beta=2.0)")
    if args.use_wiou:
        print("[LOSS] WIoU v3 Loss (dynamic non-monotonic focusing)")
    if args.use_focal_wiou:
        print("[LOSS] Focal-WIoU Loss")
    if args.use_gcd:
        print("[LOSS] Gaussian Combined Distance box loss (pure GCD box term; DFL retained)")
    if args.iou_type != "ciou":
        print(f"[LOSS] IoU variant: {args.iou_type.upper()}")
    if args.ordinal:
        print("[LOSS] Ordinal Soft Labels for ordered ripeness classification")
    if args.use_oql:
        print(f"[LOSS] Expected-stage SmoothL1 ordinal loss (lambda={args.oql_lambda}, "
              f"expect={args.oql_expect_weight}, "
              f"kl={args.oql_kl_weight}, neighbor={args.oql_neighbor}, tau={args.oql_tau})")
    if args.use_qcl:
        print(f"[LOSS] Quality Consistency Loss (lambda={args.qcl_lambda}, type={args.qcl_type}, "
              f"detach_iou={args.qcl_detach_iou}, min_iou={args.qcl_min_iou})")


def _shutdown_loader(loader):
    """Best-effort shutdown of persistent DataLoader workers before process exit."""
    # Do not use ``a or b`` here. PyTorch DataLoader iterators implement
    # ``__len__``; testing their truth value can therefore call __len__ and
    # fail for Ultralytics' infinite loader (_RepeatSampler has no len()).
    iterator = getattr(loader, "iterator", None)
    if iterator is None:
        iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception as exc:
            # Resource release is best-effort and must never turn an otherwise
            # completed training run into exit code 1.
            print(f"[RESOURCE WARNING] DataLoader shutdown skipped: {exc}")


def _release_training_resources(model):
    """Release loader workers, shared memory and CUDA cache after each experiment."""
    trainer = getattr(model, "trainer", None) if model is not None else None
    if trainer is not None:
        for loader_name in ("train_loader", "test_loader"):
            try:
                _shutdown_loader(getattr(trainer, loader_name, None))
            except Exception as exc:
                print(f"[RESOURCE WARNING] {loader_name} cleanup skipped: {exc}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def _load_mapped_pretrained(model: YOLO, weights: str, mapping_name: str) -> YOLO:
    """Load shape-compatible tensors after explicitly remapping base-model layer indices."""
    mappings = {
        "peanet_yolov5d_repro": {
            **{index: index + 1 for index in range(0, 19)},
            19: 21,
            20: 22,
            21: 24,
            22: 25,
            23: 26,
            24: 28,
        },
        "fpdw_yolov8_repro": {
            **{index: index for index in range(0, 16)},
            16: 17,
            17: 18,
            18: 19,
            19: 21,
            20: 22,
            21: 23,
            22: 25,
        },
        # The released Edge graph changes all feature blocks and the entire
        # neck/head. Load only shape-compatible YOLO11s backbone tensors.
        "edge_yolov11_repro": {index: index for index in range(0, 11)},
        # GPC replaces downsampling and C2f internals, but C2f-PC retains the
        # compatible CSP projections in the YOLOv8n backbone.
        "gpc_yolov8_repro": {index: index for index in range(0, 10)},
        # FSDH inserts SHAB after YOLO11 backbone stages 4, 6 and 10. Remap
        # every unchanged YOLO11n layer by semantic position; FSDConv and SHAB
        # remain seed-controlled random initializations by design.
        "fsdh_yolo11n": {
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 6,
            6: 7,
            7: 9,
            8: 10,
            9: 11,
            10: 12,
            13: 16,
            16: 19,
            17: 20,
            19: 22,
            20: 23,
            22: 25,
            23: 26,
        },
    }
    source = YOLO(weights).model.state_dict()
    target = model.model.state_dict()
    layer_map = mappings[mapping_name]
    compatible = {}
    for key, value in source.items():
        match = re.match(r"model\.(\d+)\.(.+)", key)
        if not match:
            continue
        source_index = int(match.group(1))
        if source_index not in layer_map:
            continue
        target_key = f"model.{layer_map[source_index]}.{match.group(2)}"
        if target_key in target and target[target_key].shape == value.shape:
            compatible[target_key] = value
    model.model.load_state_dict(compatible, strict=False)
    matched_numel = sum(target[key].numel() for key in compatible)
    total_numel = sum(value.numel() for value in target.values())
    model.paper_init_summary = {
        "mode": "semantic_layer_mapping",
        "weights": str(Path(weights).resolve()),
        "weights_map": mapping_name,
        "matched_state_items": len(compatible),
        "target_state_items": len(target),
        "matched_state_elements": matched_numel,
        "target_state_elements": total_numel,
        "matched_state_elements_percent": 100.0 * matched_numel / total_numel,
    }
    print(
        f"[INFO] Mapped {len(compatible)}/{len(target)} tensors from {weights} ({mapping_name}); "
        f"state elements={matched_numel}/{total_numel} ({100.0 * matched_numel / total_numel:.2f}%)"
    )
    return model


def _native_transfer_summary(model: YOLO, weights: str) -> dict[str, object]:
    """Describe Ultralytics' native same-name/same-shape transfer without changing it."""
    checkpoint = getattr(model, "ckpt", None)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Cannot audit native pretrained transfer from {weights}: checkpoint metadata is missing")
    source_model = checkpoint.get("ema") or checkpoint.get("model")
    if source_model is None:
        raise RuntimeError(f"Cannot audit native pretrained transfer from {weights}: source model is missing")

    source = source_model.float().state_dict()
    target = model.model.state_dict()
    compatible = intersect_dicts(source, target)
    matched_items = len(compatible)
    matched_numel = sum(target[key].numel() for key in compatible)

    # Mirror BaseModel.load()'s optional partial first-convolution transfer.
    first_conv = "model.0.conv.weight"
    if first_conv not in compatible and first_conv in source and first_conv in target:
        c1, c2, h, w = target[first_conv].shape
        sc1, sc2, sh, sw = source[first_conv].shape
        if sh == h and sw == w:
            matched_items += 1
            matched_numel += min(c1, sc1) * min(c2, sc2) * h * w

    total_numel = sum(value.numel() for value in target.values())
    return {
        "mode": "native_name_shape_transfer",
        "weights": str(Path(weights).resolve()),
        "weights_map": "none",
        "matched_state_items": matched_items,
        "target_state_items": len(target),
        "matched_state_elements": matched_numel,
        "target_state_elements": total_numel,
        "matched_state_elements_percent": 100.0 * matched_numel / total_numel,
    }


def main():
    args = parse_args()
    configure_loss_environment(args)
    print_run_banner(args)

    model = None
    try:
        # YOLO(args.model) constructs custom-module parameters before Trainer
        # initializes its seed. Seed here as well so FSDConv/SHAB initialization
        # is determined by --seed and can be reproduced exactly.
        init_seeds(args.seed + 1, deterministic=True)
        model = YOLO(args.model)
        initialization_summary = {
            "schema": "paper16_training_initialization_v1",
            "model_argument": str(Path(args.model).resolve()),
            "seed_argument": args.seed,
            "pre_model_seed": args.seed + 1,
            "deterministic": True,
            "mode": "checkpoint_model" if str(args.model).lower().endswith(".pt") else "model_yaml",
            "weights": None,
            "weights_map": "none",
        }
        if args.weights and os.path.exists(args.weights):
            if args.weights_map == "none":
                model = model.load(args.weights)
                initialization_summary.update(_native_transfer_summary(model, args.weights))
                print(f"[INFO] Loaded pretrained weights from {args.weights}")
            else:
                model = _load_mapped_pretrained(model, args.weights, args.weights_map)
                initialization_summary.update(model.paper_init_summary)

        model.train(
            data=args.data,
            task="detect",
            cache=False,
            imgsz=args.imgsz,
            epochs=args.epochs,
            single_cls=False,
            batch=args.batch,
            close_mosaic=0,
            workers=args.workers,
            device="0",
            optimizer="SGD",
            amp=True,
            patience=args.patience,
            seed=args.seed,
            fraction=args.fraction,
            deterministic=True,
            project=args.project,
            name=args.name,
            exist_ok=False,
            resume=False,
            use_focal_loss=args.use_focal,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
        )

        trainer = getattr(model, "trainer", None)
        validator = getattr(trainer, "validator", None)
        final_metrics = getattr(validator, "metrics", None) or getattr(model, "metrics", None)
        exp_dir = getattr(trainer, "save_dir", os.path.join(args.project, args.name))
        initialization_summary["run_directory"] = str(Path(exp_dir).resolve())
        Path(exp_dir, "initialization_summary.json").write_text(
            json.dumps(initialization_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        write_summary(
            exp_dir=exp_dir,
            data_yaml=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device="0",
            patience=args.patience,
            total_epochs=args.epochs,
            yolo_model=model,
            metrics=final_metrics,
        )
    finally:
        _release_training_resources(model)


if __name__ == "__main__":
    main()
