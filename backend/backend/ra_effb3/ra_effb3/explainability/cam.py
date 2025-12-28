from __future__ import annotations

import os
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path

from ra_effb3.model.layers import EfficientNetPreprocess, ReduceMeanLayer, ReduceMaxLayer, EfficientNetB3Block


def _load(model_path: str) -> tf.keras.Model:
    custom_objects = {
        "EfficientNetPreprocess": EfficientNetPreprocess,
        "ReduceMeanLayer": ReduceMeanLayer,
        "ReduceMaxLayer": ReduceMaxLayer,
        "EfficientNetB3Block": EfficientNetB3Block,
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)


def make_heatmap(img_array, model, last_conv_layer_name: str, pred_index=None):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay(bgr_img, heatmap, alpha=0.35):
    heatmap = cv2.resize(heatmap, (bgr_img.shape[1], bgr_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr_img, 1 - alpha, heatmap_color, alpha, 0)


def run_gradcam_on_folder(
    model_path: str,
    input_folder: str,
    output_folder: str,
    img_size: int,
    class_names: list[str],
    last_conv_layer_name: str = "top_conv",
):
    model = _load(model_path)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG")
    files = [f for f in os.listdir(input_folder) if f.endswith(exts)]
    files.sort()

    for fn in files:
        src = os.path.join(input_folder, fn)
        bgr = cv2.imread(src)
        if bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)

        x = np.expand_dims(rgb, axis=0).astype(np.float32)
        preds = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_label = class_names[pred_idx]
        conf = float(preds[pred_idx])

        heatmap = make_heatmap(x, model, last_conv_layer_name, pred_index=pred_idx)
        overlay_img = overlay(cv2.resize(bgr, (img_size, img_size)), heatmap, alpha=0.40)

        out_name = f"{Path(fn).stem}_pred-{pred_label}_conf-{conf:.3f}.jpg"
        cv2.imwrite(os.path.join(output_folder, out_name), overlay_img)
