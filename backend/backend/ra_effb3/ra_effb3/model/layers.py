from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import preprocess_input


class EfficientNetPreprocess(layers.Layer):
    def call(self, x):
        x = tf.cast(x, tf.float32)
        return preprocess_input(x)

    def get_config(self):
        return super().get_config()


class ReduceMeanLayer(layers.Layer):
    def call(self, x):
        return tf.reduce_mean(x, axis=-1, keepdims=True)

    def get_config(self):
        return super().get_config()


class ReduceMaxLayer(layers.Layer):
    def call(self, x):
        return tf.reduce_max(x, axis=-1, keepdims=True)

    def get_config(self):
        return super().get_config()


def residual_attention_block(x, filters: int, name: str = "ra"):
    shortcut = x

    y = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(x)
    y = layers.BatchNormalization(name=f"{name}_bn")(y)
    y = layers.Activation("swish", name=f"{name}_act")(y)

    gap = layers.GlobalAveragePooling2D(name=f"{name}_gap")(y)
    ca = layers.Dense(max(filters // 8, 8), activation="swish", name=f"{name}_ca_fc1")(gap)
    ca = layers.Dense(filters, activation="sigmoid", name=f"{name}_ca_fc2")(ca)
    ca = layers.Reshape((1, 1, filters), name=f"{name}_ca_reshape")(ca)
    y = layers.Multiply(name=f"{name}_ca_mul")([y, ca])

    mean_map = ReduceMeanLayer(name=f"{name}_sa_mean")(y)
    max_map = ReduceMaxLayer(name=f"{name}_sa_max")(y)
    sa = layers.Concatenate(name=f"{name}_sa_cat")([mean_map, max_map])
    sa = layers.Conv2D(1, 7, padding="same", activation="sigmoid", name=f"{name}_sa_conv")(sa)
    y = layers.Multiply(name=f"{name}_sa_mul")([y, sa])

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_proj")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    out = layers.Add(name=f"{name}_add")([shortcut, y])
    out = layers.Activation("swish", name=f"{name}_out_act")(out)
    return out


class EfficientNetB3Block(layers.Layer):
    def __init__(self, trainable_base: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.base = tf.keras.applications.EfficientNetB3(
            include_top=False,
            weights="imagenet",
            input_shape=(300, 300, 3),
        )
        self.base.trainable = bool(trainable_base)

    def call(self, x, training=None):
        return self.base(x, training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"trainable_base": self.base.trainable})
        return cfg
