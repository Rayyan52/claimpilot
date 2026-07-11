"""
Trains the damage-severity CNN (MobileNetV2 transfer learning) on the
Car Damage Severity Dataset, saves the model + class index mapping, and
runs one Grad-CAM sanity check image.

Run: python scripts/train_cnn.py
"""
import json
import os

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from cnn_utils import make_gradcam_heatmap, overlay_heatmap

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "car_damage")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def find_split_dirs(root):
    """Locate the training/validation directories regardless of nesting depth."""
    train_dir = val_dir = None
    for dirpath, dirnames, _ in os.walk(root):
        lower = [d.lower() for d in dirnames]
        for d, low in zip(dirnames, lower):
            full = os.path.join(dirpath, d)
            if low in ("training", "train") and any(
                os.path.isdir(os.path.join(full, c)) for c in os.listdir(full)
            ):
                train_dir = full
            elif low in ("validation", "valid", "val") and any(
                os.path.isdir(os.path.join(full, c)) for c in os.listdir(full)
            ):
                val_dir = full
    if not train_dir or not val_dir:
        raise FileNotFoundError(
            f"Could not find training/validation folders under {root}. "
            f"Found: {os.listdir(root)}"
        )
    return train_dir, val_dir


def build_model(num_classes):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    train_dir, val_dir = find_split_dirs(DATA_DIR)
    print(f"Training dir: {train_dir}")
    print(f"Validation dir: {val_dir}")

    train_gen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=15,
        zoom_range=0.15,
        horizontal_flip=True,
    ).flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")

    val_gen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    ).flow_from_directory(val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")

    class_indices = train_gen.class_indices  # e.g. {'minor': 0, 'moderate': 1, 'severe': 2}
    print("Classes:", class_indices)

    model = build_model(num_classes=len(class_indices))
    model.summary()

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

    model.save(os.path.join(MODELS_DIR, "damage_cnn.h5"))
    with open(os.path.join(MODELS_DIR, "damage_cnn_classes.json"), "w") as f:
        json.dump({v: k for k, v in class_indices.items()}, f)
    print("Saved model and class mapping to models/")

    # Grad-CAM sanity check on one validation image
    val_gen.reset()
    sample_batch, _ = next(val_gen)
    sample_img = sample_batch[0:1]
    heatmap, pred_idx = make_gradcam_heatmap(sample_img, model)

    from PIL import Image
    import numpy as np

    denorm = ((sample_batch[0] + 1) * 127.5).astype("uint8")  # undo mobilenet_v2 preprocess_input
    pil_img = Image.fromarray(denorm)
    overlay = overlay_heatmap(pil_img, heatmap)
    overlay_path = os.path.join(MODELS_DIR, "gradcam_sanity_check.png")
    overlay.save(overlay_path)
    idx_to_class = {v: k for k, v in class_indices.items()}
    print(f"Grad-CAM sanity check saved to {overlay_path} (predicted: {idx_to_class[pred_idx]})")


if __name__ == "__main__":
    main()
