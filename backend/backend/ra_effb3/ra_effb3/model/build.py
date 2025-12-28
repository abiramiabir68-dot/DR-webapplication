from __future__ import annotations

from tensorflow.keras import layers, models, regularizers

from .layers import EfficientNetPreprocess, EfficientNetB3Block, residual_attention_block


def build_model(
    input_shape=(300, 300, 3),
    num_classes=5,
    train_base=False,
    aug_layer=None,
):
    inp = layers.Input(shape=input_shape, name="image")

    x = inp
    if aug_layer is not None:
        x = aug_layer(x)

    x = EfficientNetPreprocess(name="preprocess")(x)

    base_block = EfficientNetB3Block(trainable_base=train_base, name="backbone")
    x = base_block(x)

    x = residual_attention_block(x, filters=256, name="ra")

    x = layers.Conv2D(256, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.35)(x)
    out = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=regularizers.l2(2e-5),
        dtype="float32",
        name="softmax",
    )(x)

    return models.Model(inputs=inp, outputs=out, name="RA_EfficientNetB3")
