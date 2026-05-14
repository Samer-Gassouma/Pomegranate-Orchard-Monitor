#!/usr/bin/env python3
"""Generate YOLO-format pseudo-labels from existing model on orchard images."""

import os
import shutil
import glob
import random
from pathlib import Path

import cv2
from ultralytics import YOLO

# Paths
ORCHARD_DIR = Path("pomegranate_orchard_dataset/Pomegranate Images Dataset")
OUTPUT_DIR = Path("datasets/pomegranate_orchard")
MODEL_PATH = "pomegranate_models/yolov8_pomegranate.pt"

# Class mapping: Turkish -> English (same IDs)
CLASS_NAMES = {
    0: "crown_damaged",
    1: "crown_good",
    2: "surface_damaged",
    3: "surface_good",
    4: "surface_burnt",
    5: "surface_cracked",
}

# Confidence threshold for pseudo-labeling
CONF_THRESHOLD = 0.15

# Train/val split
TRAIN_RATIO = 0.8


def clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def generate_labels(model, image_paths, images_out, labels_out):
    """Run inference and save YOLO-format .txt labels."""
    labeled_count = 0
    total_boxes = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Skip (unreadable): {img_path.name}")
            continue

        h, w = img.shape[:2]

        # Run inference
        results = model(str(img_path), conf=CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            # Copy image even if no labels (negative sample)
            shutil.copy(img_path, images_out / img_path.name)
            # Create empty label file
            (labels_out / f"{img_path.stem}.txt").write_text("")
            continue

        lines = []
        for box in boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Convert to YOLO format (normalized center x, center y, width, height)
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            # Clamp to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            bw = max(0.0, min(1.0, bw))
            bh = max(0.0, min(1.0, bh))

            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        # Copy image and write label
        shutil.copy(img_path, images_out / img_path.name)
        (labels_out / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n")

        labeled_count += 1
        total_boxes += len(lines)

    return labeled_count, total_boxes


def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    print(f"Model classes: {model.names}\n")

    # Collect valid images
    image_paths = []
    for split in ["train", "val", "test"]:
        split_dir = ORCHARD_DIR / split
        if split_dir.exists():
            image_paths.extend(sorted(split_dir.glob("*.jpg")))

    image_paths = [p for p in image_paths if cv2.imread(str(p)) is not None]
    print(f"Total valid images: {len(image_paths)}")

    # Shuffle and split
    random.seed(42)
    random.shuffle(image_paths)
    split_idx = int(len(image_paths) * TRAIN_RATIO)
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]

    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}\n")

    # Prepare output directories
    clean_dir(OUTPUT_DIR)
    (OUTPUT_DIR / "images" / "train").mkdir(parents=True)
    (OUTPUT_DIR / "images" / "val").mkdir(parents=True)
    (OUTPUT_DIR / "labels" / "train").mkdir(parents=True)
    (OUTPUT_DIR / "labels" / "val").mkdir(parents=True)

    # Generate labels
    print("Generating train labels...")
    t_count, t_boxes = generate_labels(
        model, train_paths,
        OUTPUT_DIR / "images" / "train",
        OUTPUT_DIR / "labels" / "train",
    )
    print(f"  {t_count}/{len(train_paths)} images labeled, {t_boxes} total boxes\n")

    print("Generating val labels...")
    v_count, v_boxes = generate_labels(
        model, val_paths,
        OUTPUT_DIR / "images" / "val",
        OUTPUT_DIR / "labels" / "val",
    )
    print(f"  {v_count}/{len(val_paths)} images labeled, {v_boxes} total boxes\n")

    # Write data.yaml
    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_content = f"""path: ../datasets/pomegranate_orchard  # dataset root (relative to training dir)
train: images/train
val: images/val
test:  # optional

nc: {len(CLASS_NAMES)}
names:
  0: crown_damaged
  1: crown_good
  2: surface_damaged
  3: surface_good
  4: surface_burnt
  5: surface_cracked
"""
    yaml_path.write_text(yaml_content)
    print(f"Written {yaml_path}")

    # Summary
    print(f"\n{'='*60}")
    print("DATASET PREPARED")
    print(f"{'='*60}")
    print(f"Location: {OUTPUT_DIR.absolute()}")
    print(f"Train images: {len(list((OUTPUT_DIR / 'images' / 'train').glob('*.jpg')))}")
    print(f"Val images:   {len(list((OUTPUT_DIR / 'images' / 'val').glob('*.jpg')))}")
    print(f"\nNext: Upload this dataset to Google Drive or Colab,")
    print(f"then run the fine-tuning notebook.")


if __name__ == "__main__":
    main()
