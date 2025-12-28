from __future__ import annotations

import tensorflow as tf

from ra_effb3.config import set_global_determinism, Paths, DataConfig
from ra_effb3.data.ds import (
    build_directory_datasets,
    compute_class_counts_from_dir,
    make_sample_weighted,
    prepare_for_training,
)
from ra_effb3.model.layers import EfficientNetPreprocess, ReduceMeanLayer, ReduceMaxLayer, EfficientNetB3Block
from ra_effb3.model.build import build_model
from ra_effb3.training.callbacks import build_callbacks, freeze_batchnorm
from ra_effb3.evaluation.metrics import evaluate


def train_finetune(paths: Paths, cfg: DataConfig, epochs: int = 15):
    set_global_determinism(cfg.SEED)
    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_directory_datasets(
        train_dir=str(paths.PREPROC_TRAIN_DIR),
        val_dir=str(paths.PREPROC_VAL_DIR),
        img_size=cfg.IMG_SIZE,
        batch_size=cfg.BATCH_SIZE,
        class_names=list(cfg.CLASS_NAMES),
        seed=cfg.SEED,
    )

    counts = compute_class_counts_from_dir(str(paths.PREPROC_TRAIN_DIR), list(cfg.CLASS_NAMES), cfg.IMG_EXTS)
    counts = tf.maximum(tf.cast(counts, tf.float32), 1.0)
    inv = 1.0 / counts
    inv = inv / tf.reduce_mean(inv)
    class_weight_vector = tf.cast(inv, tf.float32)

    train_ds = make_sample_weighted(train_ds, class_weight_vector)
    train_ds = prepare_for_training(train_ds, shuffle=True, seed=cfg.SEED)
    val_ds = prepare_for_training(val_ds, shuffle=False, seed=cfg.SEED)

    custom_objects = {
        "EfficientNetPreprocess": EfficientNetPreprocess,
        "ReduceMeanLayer": ReduceMeanLayer,
        "ReduceMaxLayer": ReduceMaxLayer,
        "EfficientNetB3Block": EfficientNetB3Block,
    }

    if tf.io.gfile.exists(str(paths.BASELINE_MODEL_PATH)):
        model = tf.keras.models.load_model(str(paths.BASELINE_MODEL_PATH), custom_objects=custom_objects, compile=False)
    else:
        model = build_model(
            input_shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3),
            num_classes=len(cfg.CLASS_NAMES),
            train_base=False,
            aug_layer=None,
        )

    # Unfreeze only high-level blocks
    backbone = None
    for layer in model.layers:
        if hasattr(layer, "base"):
            backbone = layer.base
            break

    if backbone is not None:
        backbone.trainable = True
        for l in backbone.layers:
            l.trainable = False
        for l in backbone.layers:
            if l.name.startswith(("block6", "block7", "top_")):
                l.trainable = True

    freeze_batchnorm(model)

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-5,
        weight_decay=1e-5,
        epsilon=1e-7,
        clipnorm=1.0,
    )
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.001)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    callbacks = build_callbacks(paths.FINE_TUNE_MODEL_PATH)

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
