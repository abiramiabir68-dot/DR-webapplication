from __future__ import annotations

import os
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def build_directory_datasets(
    train_dir: str,
    val_dir: str,
    img_size: int,
    batch_size: int,
    class_names: list[str],
    seed: int,
):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        class_names=class_names,
        shuffle=True,
        seed=seed,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        class_names=class_names,
        shuffle=False,
    )

    return train_ds, val_ds


def compute_class_counts_from_dir(train_dir: str, class_names: list[str], img_exts: tuple[str, ...]) -> np.ndarray:
    counts = []
    for cname in class_names:
        class_dir = os.path.join(train_dir, cname)
        if not os.path.isdir(class_dir):
            counts.append(0)
            continue
        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(tuple(e.lower() for e in img_exts))
        ]
        counts.append(len(files))
    return np.array(counts, dtype=np.int64)


def make_sample_weighted(ds: tf.data.Dataset, class_weight_vector: tf.Tensor) -> tf.data.Dataset:
    def add_sw(images, labels):
        class_ids = tf.argmax(labels, axis=-1, output_type=tf.int32)
        sw = tf.gather(class_weight_vector, class_ids)
        sw = tf.cast(sw, tf.float32)
        return images, labels, sw
    return ds.map(add_sw, num_parallel_calls=AUTOTUNE)


def prepare_for_training(ds: tf.data.Dataset, shuffle: bool, seed: int) -> tf.data.Dataset:
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(1024, seed=seed, reshuffle_each_iteration=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds
