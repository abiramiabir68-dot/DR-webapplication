from __future__ import annotations

from pathlib import Path
import tensorflow as tf


def build_callbacks(model_path: str | Path):
    model_path = str(model_path)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )

    plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=3,
        min_delta=3e-4,
        cooldown=1,
        min_lr=1e-6,
        verbose=1,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=6,
        min_delta=3e-4,
        restore_best_weights=True,
        verbose=1,
    )

    return [checkpoint, plateau, early_stop]


def freeze_batchnorm(model: tf.keras.Model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        if hasattr(layer, "layers"):
            freeze_batchnorm(layer)
        if hasattr(layer, "base"):
            freeze_batchnorm(layer.base)
