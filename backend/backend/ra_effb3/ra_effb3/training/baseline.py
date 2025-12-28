from __future__ import annotations

import numpy as np
import tensorflow as tf

from ra_effb3.config import set_global_determinism, Paths, DataConfig
from ra_effb3.preprocessing.clahe import preprocess_from_csv
from ra_effb3.data.ds import (
    build_directory_datasets,
    compute_class_counts_from_dir,
    make_sample_weighted,
    prepare_for_training,
)
from ra_effb3.model.build import build_model
from ra_effb3.training.callbacks import build_callbacks, freeze_batchnorm
from ra_effb3.evaluation.metrics import evaluate


def train_baseline(paths: Paths, cfg: DataConfig, epochs: int = 15):
    set_global_determinism(cfg.SEED)
    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build preprocessed folders (comment out if you already did this once)
    preprocess_from_csv(paths.TRAINING_CSV, paths.TRAIN_IMG_PATH, str(paths.PREPROC_TRAIN_DIR), cfg.CLASS_MAP, cfg.IMG_SIZE, cfg.IMG_EXTS)
    preprocess_from_csv(paths.VALIDATING_CSV, paths.VAL_IMG_PATH, str(paths.PREPROC_VAL_DIR), cfg.CLASS_MAP, cfg.IMG_SIZE, cfg.IMG_EXTS)

    train_ds, val_ds = build_directory_datasets(
        train_dir=str(paths.PREPROC_TRAIN_DIR),
        val_dir=str(paths.PREPROC_VAL_DIR),
        img_size=cfg.IMG_SIZE,
        batch_size=cfg.BATCH_SIZE,
        class_names=list(cfg.CLASS_NAMES),
        seed=cfg.SEED,
    )

    # Simple inverse-frequency sample weights
    counts = compute_class_counts_from_dir(str(paths.PREPROC_TRAIN_DIR), list(cfg.CLASS_NAMES), cfg.IMG_EXTS)
    counts = np.maximum(counts, 1)
    inv = 1.0 / counts.astype(np.float32)
    inv = inv / np.mean(inv)
    class_weight_vector = tf.constant(inv, dtype=tf.float32)

    train_ds = make_sample_weighted(train_ds, class_weight_vector)
    train_ds = prepare_for_training(train_ds, shuffle=True, seed=cfg.SEED)
    val_ds = prepare_for_training(val_ds, shuffle=False, seed=cfg.SEED)

    aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="aug",
    )

    model = build_model(
        input_shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3),
        num_classes=len(cfg.CLASS_NAMES),
        train_base=False,
        aug_layer=aug,
    )

    freeze_batchnorm(model)

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=3e-5,
        weight_decay=1e-5,
        epsilon=1e-7,
        clipnorm=1.0,
    )
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.002)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    callbacks = build_callbacks(paths.BASELINE_MODEL_PATH)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    result = evaluate(model, val_ds, class_names=list(cfg.CLASS_NAMES))
    print("\nFINAL CLASSIFICATION REPORT (VAL)\n")
    print(result["report_text"])
    print("\nCONFUSION MATRIX (VAL)\n")
    print(result["confusion_matrix"])
