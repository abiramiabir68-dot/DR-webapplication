"""CLAHE preprocessing.
Reads CSVs with columns: id_code, diagnosis
Writes images into out_dir/<class_name>/ for use with image_dataset_from_directory.
"""
from __future__ import annotations

from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

from ra_effb3.fs import ensure_dir


def apply_clahe_bgr(image_bgr, img_size: tuple[int, int]):
    image_bgr = cv2.resize(image_bgr, img_size, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    lab2 = cv2.merge([l2, a, b])
    enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return enhanced


def preprocess_from_csv(
    csv_path: str,
    raw_images_dir: str,
    out_dir: str,
    class_map: dict[int, str],
    img_size: int = 300,
    img_exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG"),
) -> None:
    df = pd.read_csv(csv_path)
    required = {"id_code", "diagnosis"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain {required}. Found: {set(df.columns)}")

    out_root = ensure_dir(out_dir)
    for _, cname in class_map.items():
        ensure_dir(out_root / cname)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"CLAHE -> {Path(out_dir).name}"):
        image_id = str(row["id_code"])
        diagnosis = int(row["diagnosis"])
        class_name = class_map[diagnosis]

        src = None
        for ext in img_exts:
            candidate = Path(raw_images_dir) / f"{image_id}{ext}"
            if candidate.exists():
                src = candidate
                break
        if src is None:
            continue

        img_bgr = cv2.imread(str(src))
        if img_bgr is None:
            continue

        enhanced = apply_clahe_bgr(img_bgr, (img_size, img_size))
        dst = Path(out_dir) / class_name / src.name
        cv2.imwrite(str(dst), enhanced)
