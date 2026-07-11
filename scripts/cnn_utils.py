"""Shared CNN inference helpers: used by train_cnn.py (sanity check) and app.py (live inference).

Assumes `model` was built with the functional API directly on top of
MobileNetV2's input/output tensors (not as a nested sub-model), so all of
MobileNetV2's layers are directly in `model.layers`.
"""
import numpy as np
import tensorflow as tf


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index.numpy())


def overlay_heatmap(pil_img, heatmap, alpha=0.4):
    import matplotlib as mpl
    from PIL import Image

    heatmap = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255)).resize(pil_img.size)
    jet_heatmap = np.array(jet_heatmap)

    superimposed = jet_heatmap * alpha + np.array(pil_img.convert("RGB"))
    superimposed = np.clip(superimposed, 0, 255).astype("uint8")
    return Image.fromarray(superimposed)
