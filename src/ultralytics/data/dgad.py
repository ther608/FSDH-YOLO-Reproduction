"""Depth-target utilities for training-only DGAD supervision on Dataset A."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

from ultralytics.utils.patches import imread


_AUGMENT_SUFFIX = re.compile(r"_aug[123]_(?:illum_color|image_quality|mild_geometry)_[^.]+$")


class DGADDepthResolver:
    """Resolve every legacy augmented RGB sample to its aligned monocular-depth target."""

    def __init__(self, source_root: str | Path, augmentation_log: str | Path):
        self.source_root = Path(source_root)
        self.augmentation_log = Path(augmentation_log)
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"DGAD source root not found: {self.source_root}")
        if not self.augmentation_log.is_file():
            raise FileNotFoundError(f"DGAD augmentation log not found: {self.augmentation_log}")
        with self.augmentation_log.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.records = {row["output_image"]: row for row in rows}
        if len(self.records) != len(rows):
            raise ValueError(f"Duplicate output_image entries in {self.augmentation_log}")

    def source_name(self, image_name: str) -> str:
        record = self.records.get(image_name)
        if record:
            return record["source_image"]
        path = Path(image_name)
        return _AUGMENT_SUFFIX.sub("", path.stem) + path.suffix

    def depth_path(self, image_name: str) -> Path:
        source_name = self.source_name(image_name)
        if "_ID-" not in source_name:
            raise ValueError(f"Cannot infer Dataset-A scene from image name: {source_name}")
        scene, tail = source_name.split("_ID-", 1)
        scene_dir = f"Image_YoloLabel_{scene}"
        return self.source_root / scene_dir / scene_dir / "depth" / "gray" / f"ID-{tail}"

    def load(self, image_name: str) -> np.ndarray:
        """Load depth and replay only the logged offline geometric transform."""
        path = self.depth_path(image_name)
        depth = imread(str(path), flags=cv2.IMREAD_GRAYSCALE)
        if depth is None:
            raise FileNotFoundError(f"DGAD depth target not found or unreadable: {path}")
        depth = depth[..., 0] if depth.ndim == 3 else depth
        record = self.records.get(image_name)
        if not record or record.get("category") != "mild_geometry":
            return depth

        operation = record["operation"]
        parameters = json.loads(record.get("parameters") or "{}")
        if operation == "horizontal_flip":
            return cv2.flip(depth, 1)

        height, width = depth.shape[:2]
        angle = float(parameters.get("angle_degrees", 0.0))
        scale = float(parameters.get("scale", 1.0))
        tx = float(parameters.get("translate_x_fraction", 0.0))
        ty = float(parameters.get("translate_y_fraction", 0.0))
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
        matrix[0, 2] += tx * width
        matrix[1, 2] += ty * height
        return cv2.warpAffine(
            depth,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def resolve(self, image_path: str | Path) -> tuple[Path, np.ndarray]:
        image_name = Path(image_path).name
        path = self.depth_path(image_name)
        return path, self.load(image_name)
