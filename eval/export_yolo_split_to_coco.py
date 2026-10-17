"""Export one YOLO-format dataset split to a traceable COCO ground-truth file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_split(data_yaml: Path, split: str) -> list[Path]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    values = data[split] if isinstance(data[split], list) else [data[split]]
    paths = [Path(value) for value in values]
    return [path if path.is_absolute() else (root / path).resolve() for path in paths]


def collect_images(paths: list[Path]) -> list[Path]:
    images: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        elif path.is_dir():
            images.extend(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
        else:
            raise FileNotFoundError(f"Split path not found: {path}")
    return sorted(images)


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / image_path.with_suffix(".txt").name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, list):
        names = {index: name for index, name in enumerate(names)}
    names = {int(index): str(name) for index, name in names.items()}
    images = collect_images(resolve_split(args.data, args.split))

    image_records, annotations = [], []
    annotation_id = 1
    for image_id, image_path in enumerate(images, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        image_records.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        target = label_path(image_path)
        raw_lines = target.read_text(encoding="utf-8").splitlines() if target.is_file() else []
        for line in dict.fromkeys(item.strip() for item in raw_lines if item.strip()):
            cls_s, cx_s, cy_s, bw_s, bh_s, *_ = line.split()
            cls = int(float(cls_s))
            if cls not in names:
                raise ValueError(f"Label {target} references class {cls}, absent from data YAML names")
            cx, cy, bw, bh = map(float, (cx_s, cy_s, bw_s, bh_s))
            box_w, box_h = bw * width, bh * height
            x = (cx - bw / 2.0) * width
            y = (cy - bh / 2.0) * height
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cls,
                    "bbox": [x, y, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    payload = {
        "info": {"source_yaml": str(args.data.resolve()), "split": args.split},
        "images": image_records,
        "annotations": annotations,
        "categories": [{"id": index, "name": names[index]} for index in sorted(names)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(images)} images and {len(annotations)} annotations to {args.out}")


if __name__ == "__main__":
    main()
