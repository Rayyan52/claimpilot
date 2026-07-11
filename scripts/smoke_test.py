"""
End-to-end smoke test of the ClaimPilot pipeline (CNN -> ANN -> Gemini -> PDF)
using the real trained artifacts, without needing Streamlit installed.

Run from the repo root (e.g. in Colab, after training):
    python scripts/smoke_test.py
"""
import glob
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.cnn_utils import make_gradcam_heatmap, overlay_heatmap
from gemini_helper import analyze_claim_text
from report_generator import build_report

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def find_sample_image():
    matches = glob.glob(
        os.path.join(os.path.dirname(__file__), "..", "data", "car_damage", "**", "*.jpg"),
        recursive=True,
    )
    if not matches:
        matches = glob.glob(
            os.path.join(os.path.dirname(__file__), "..", "data", "car_damage", "**", "*.jpeg"),
            recursive=True,
        )
    if not matches:
        raise FileNotFoundError("No sample image found under data/car_damage")
    return matches[0]


def main():
    cnn_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "damage_cnn.h5"))
    with open(os.path.join(MODELS_DIR, "damage_cnn_classes.json")) as f:
        cnn_classes = {int(k): v for k, v in json.load(f).items()}

    ann_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "risk_ann.h5"))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "risk_preprocessor.pkl"))
    with open(os.path.join(MODELS_DIR, "risk_form_schema.json")) as f:
        form_schema = json.load(f)

    sample_image_path = find_sample_image()
    print(f"Using sample image: {sample_image_path}")
    pil_img = Image.open(sample_image_path).convert("RGB").resize((224, 224))
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.expand_dims(np.array(pil_img).astype("float32"), axis=0)
    )
    heatmap, pred_idx = make_gradcam_heatmap(img_array, cnn_model)
    probs = cnn_model.predict(img_array, verbose=0)[0]
    severity = cnn_classes[pred_idx]
    confidence = float(probs[pred_idx])
    overlay_img = overlay_heatmap(pil_img, heatmap)
    print(f"CNN prediction: {severity} ({confidence:.0%})")

    claim_fields = {}
    for col in form_schema["numeric_cols"]:
        lo, hi = form_schema["numeric_ranges"][col]
        claim_fields[col] = (lo + hi) / 2
    for col in form_schema["categorical_cols"]:
        claim_fields[col] = form_schema["categorical_options"][col][0]
    row_df = pd.DataFrame([claim_fields])
    X_t = preprocessor.transform(row_df).astype("float32")
    risk_score = float(ann_model.predict(X_t, verbose=0).ravel()[0])
    print(f"ANN risk score: {risk_score:.0%}")

    description = "Rear-ended at a red light, bumper is dented and the taillight is cracked."
    gemini_result = analyze_claim_text(
        description, {"severity": severity, "confidence": confidence}, {"risk_score": risk_score}
    )
    print("Gemini result:", json.dumps(gemini_result, indent=2))

    result = {
        "severity": severity,
        "confidence": confidence,
        "overlay_img": overlay_img,
        "risk_score": risk_score,
        "gemini": gemini_result,
        "description": description,
        "claim_fields": claim_fields,
    }
    decision_record = {
        "decision": "Modify",
        "comment": "Smoke test run - reduced payout pending manual photo review.",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    pdf_bytes = build_report(result, decision_record)
    out_path = os.path.join(MODELS_DIR, "..", "smoke_test_report.pdf")
    with open(out_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"PDF report written to {out_path} ({len(pdf_bytes)} bytes)")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
