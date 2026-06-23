"""
Nutrition5k dataset loader.

Directory layout expected (after download_dataset.sh):
  data/nutrition5k/
    dish_ids/splits/
      rgb_test_ids.txt
      rgb_train_ids.txt
    metadata/
      dish_metadata_cafe1.csv
      dish_metadata_cafe2.csv
    imagery/
      side_angles/
        {dish_id}/
          camera_A/
            {frame_files}.jpg
          camera_B/ ...
          camera_C/ ...
          camera_D/ ...
"""

from __future__ import annotations
import csv
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Dish:
    dish_id: str
    ingredients: list[str]          # ground-truth ingredient names (lowercased)
    total_calories: float
    total_mass_g: float


def _parse_metadata_csv(path: Path) -> dict[str, Dish]:
    # Actual CSV format (no header):
    # dish_id, total_cal, total_mass, total_fat, total_carb, total_protein,
    # ingr_id, ingr_name, ingr_grams, ingr_cal, ingr_fat, ingr_carb, ingr_protein,
    # ingr_id, ingr_name, ...  (7 fields per ingredient, repeating)
    dishes: dict[str, Dish] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            dish_id = row[0].strip()
            try:
                total_calories = float(row[1])
                total_mass = float(row[2])
            except (IndexError, ValueError):
                continue

            ingredients: list[str] = []
            # Ingredient groups start at index 6: (id, name, grams, cal, fat, carb, protein)
            ingr_start = 6
            fields_per_ingr = 7
            i = ingr_start
            while i + 1 < len(row):
                name = row[i + 1].strip().lower()
                if name:
                    ingredients.append(name)
                i += fields_per_ingr

            dishes[dish_id] = Dish(
                dish_id=dish_id,
                ingredients=ingredients,
                total_calories=total_calories,
                total_mass_g=total_mass,
            )
    return dishes


def load_dishes(data_dir: str | Path) -> dict[str, Dish]:
    """Load all dish metadata from both cafe CSVs. Returns dish_id -> Dish."""
    data_dir = Path(data_dir)
    metadata_dir = data_dir / "metadata"
    dishes: dict[str, Dish] = {}
    for csv_file in sorted(metadata_dir.glob("dish_metadata_cafe*.csv")):
        dishes.update(_parse_metadata_csv(csv_file))
    return dishes


def load_split_ids(data_dir: str | Path, split: str = "test", prefix: str = "rgb") -> list[str]:
    """Load dish IDs for a split.
    split: 'test' or 'train'
    prefix: 'rgb' (side-angle dishes) or 'depth' (overhead dishes)
    """
    data_dir = Path(data_dir)
    split_file = data_dir / "dish_ids" / "splits" / f"{prefix}_{split}_ids.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    return ids


def find_overhead_rgb(data_dir: str | Path, dish_id: str) -> Path | None:
    """Find the overhead RGB image. Checks both flat ({dish_id}.png) and
    nested ({dish_id}/rgb.png) layouts."""
    base = Path(data_dir) / "imagery" / "realsense_overhead"
    flat = base / f"{dish_id}.png"
    if flat.exists():
        return flat
    nested = base / dish_id / "rgb.png"
    if nested.exists():
        return nested
    return None


def find_frame(
    data_dir: str | Path,
    dish_id: str,
    camera: str = "A",
    frame_index: int = 10,
) -> Path | None:
    """
    Find a specific frame image for a dish and camera.

    PMC13092701 uses the 10th frame (frame_index=10) of each side camera video.
    Frame filenames vary by source — tries both zero-padded and unpadded variants.
    Returns None if no image found.
    """
    data_dir = Path(data_dir)
    cam_dir = data_dir / "imagery" / "side_angles" / dish_id / f"camera_{camera}"
    if not cam_dir.exists():
        return None

    # Try common naming conventions
    candidates = [
        cam_dir / f"frame_{frame_index:03d}.jpg",
        cam_dir / f"frame_{frame_index:04d}.jpg",
        cam_dir / f"{frame_index:03d}.jpg",
        cam_dir / f"rgb_{frame_index:04d}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fall back: sort all images and take the frame_index-th one
    images = sorted(cam_dir.glob("*.jpg")) + sorted(cam_dir.glob("*.png"))
    if frame_index < len(images):
        return images[frame_index]

    # Last resort: any image in the directory
    if images:
        return images[0]

    return None


def find_all_camera_frames(
    data_dir: str | Path,
    dish_id: str,
    cameras: list[str] | None = None,
    frame_index: int = 10,
) -> dict[str, Path]:
    """
    Returns {camera_label: frame_path} for all available cameras.
    cameras defaults to ['A', 'B', 'C', 'D'].
    """
    if cameras is None:
        cameras = ["A", "B", "C", "D"]
    result: dict[str, Path] = {}
    for cam in cameras:
        frame = find_frame(data_dir, dish_id, camera=cam, frame_index=frame_index)
        if frame:
            result[cam] = frame
    return result
