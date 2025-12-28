"""Central configuration (edit paths here only)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class Paths:
    # Base directory in Drive (change if needed)
    BASE_PATH: str = "/content/drive/MyDrive/Aptos 2019"

    # Raw image folders (change if needed)
    TRAIN_IMG_PATH: str = "/content/drive/MyDrive/Aptos 2019/train_images/train_images"
    VAL_IMG_PATH: str   = "/content/drive/MyDrive/Aptos 2019/val_images/val_images"
    TEST_IMG_PATH: str  = "/content/drive/MyDrive/Aptos 2019/test_images/test_images"

    # CSVs (change if needed)
    TRAINING_CSV: str = "/content/drive/MyDrive/Aptos 2019/train_1.csv"
    VALIDATING_CSV: str = "/content/drive/MyDrive/Aptos 2019/valid.csv"
    TESTING_CSV: str = "/content/drive/MyDrive/Aptos 2019/test.csv"

    # CLAHE output root (written as class subfolders for image_dataset_from_directory)
    PREPROC_DIR: Path = field(default_factory=lambda: Path("/content/drive/MyDrive/Aptos 2019/preprocessed"))
    PREPROC_TRAIN_DIR: Path = field(init=False)
    PREPROC_VAL_DIR: Path = field(init=False)
    PREPROC_TEST_DIR: Path = field(init=False)

    # Models + outputs
    MODELS_DIR: Path = field(default_factory=lambda: Path("/content/drive/MyDrive/Aptos 2019/models"))
    OUTPUT_DIR: Path = field(default_factory=lambda: Path("/content/drive/MyDrive/Aptos 2019/outputs"))

    BASELINE_MODEL_PATH: Path = field(init=False)
    FINE_TUNE_MODEL_PATH: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "PREPROC_TRAIN_DIR", self.PREPROC_DIR / "train")
        object.__setattr__(self, "PREPROC_VAL_DIR", self.PREPROC_DIR / "val")
        object.__setattr__(self, "PREPROC_TEST_DIR", self.PREPROC_DIR / "test")

        object.__setattr__(self, "BASELINE_MODEL_PATH", self.MODELS_DIR / "best_ra_baseline.keras")
        object.__setattr__(self, "FINE_TUNE_MODEL_PATH", self.MODELS_DIR / "best_ra_finetune.keras")


@dataclass(frozen=True)
class DataConfig:
    IMG_SIZE: int = 300
    BATCH_SIZE: int = 24
    SEED: int = 42
    IMG_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG")

    CLASS_NAMES: tuple[str, ...] = (
        "No_DR",
        "Mild",
        "Moderate",
        "Severe",
        "Proliferative_DR",
    )

    CLASS_MAP: dict[int, str] = field(default_factory=dict)
    INV_CLASS_MAP: dict[str, int] = field(default_factory=dict)


def make_data_config() -> DataConfig:
    cfg = DataConfig()
    class_map = {i: n for i, n in enumerate(cfg.CLASS_NAMES)}
    inv = {v: k for k, v in class_map.items()}
    object.__setattr__(cfg, "CLASS_MAP", class_map)
    object.__setattr__(cfg, "INV_CLASS_MAP", inv)
    return cfg


def set_global_determinism(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
