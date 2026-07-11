"""
ClaimPilot - AI Co-Pilot for Insurance Claim Processing.

Pipeline: CNN (damage severity + Grad-CAM) + ANN (fraud/risk score +
feature importance) + Gemini (text reasoning/consistency check) ->
Human-in-the-Loop decision -> downloadable PDF report.
"""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

import joblib

# Streamlit Cloud secrets don't auto-populate os.environ like a local .env
# file does; bridge them here so gemini_helper's os.environ.get(...) works
# the same way in both environments. Locally there's no secrets.toml at all,
# so guard against that raising instead of just returning empty.
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

from gemini_helper import analyze_claim_text
from report_generator import build_report
from scripts.cnn_utils import make_gradcam_heatmap, overlay_heatmap

MODELS_DIR = "models"
LOG_PATH = "decision_log.csv"

FIELD_GROUPS = {
    "👤 Policy & Customer": [
        "months_as_customer", "age", "insured_sex",
        "policy_deductable", "policy_annual_premium", "umbrella_limit",
        "capital-gains", "capital-loss",
    ],
    "🚦 Incident Details": [
        "incident_type", "incident_severity", "incident_hour_of_the_day",
        "number_of_vehicles_involved", "authorities_contacted",
        "property_damage", "police_report_available", "bodily_injuries", "witnesses",
    ],
    "💰 Claim Amounts": [
        "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim",
    ],
}

st.set_page_config(page_title="ClaimPilot", page_icon="🚗", layout="wide")


def field_label(col: str) -> str:
    return col.replace("_", " ").replace("-", " ").title()


@st.cache_resource
def load_models():
    cnn_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "damage_cnn.h5"))
    with open(os.path.join(MODELS_DIR, "damage_cnn_classes.json")) as f:
        cnn_classes = {int(k): v for k, v in json.load(f).items()}

    ann_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "risk_ann.h5"))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "risk_preprocessor.pkl"))

    with open(os.path.join(MODELS_DIR, "risk_form_schema.json")) as f:
        form_schema = json.load(f)
    with open(os.path.join(MODELS_DIR, "risk_feature_importance.json")) as f:
        feature_importance = json.load(f)

    return cnn_model, cnn_classes, ann_model, preprocessor, form_schema, feature_importance


cnn_model, cnn_classes, ann_model, preprocessor, form_schema, feature_importance = load_models()

st.title("🚗 ClaimPilot")
st.caption(
    "AI Co-Pilot for Insurance Claims — upload a damage photo and claim details. "
    "AI analyzes damage severity, fraud risk, and description consistency; "
    "a human adjuster makes the final call."
)
st.divider()

if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

st.subheader("1️⃣ Submit Claim")

with st.form("claim_form"):
    st.markdown("**📸 Damage Photo & Description**")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        uploaded_image = st.file_uploader("Damage photo", type=["jpg", "jpeg", "png"])
    with pcol2:
        description = st.text_area(
            "Claim description (free text)",
            height=120,
            placeholder="Describe what happened in the incident...",
        )

    numeric_inputs = {}
    categorical_inputs = {}
    for group_name, cols in FIELD_GROUPS.items():
        with st.expander(group_name, expanded=True):
            grid = st.columns(3)
            for i, col in enumerate(cols):
                with grid[i % 3]:
                    if col in form_schema["numeric_cols"]:
                        lo, hi = form_schema["numeric_ranges"][col]
                        hi = hi if hi > lo else lo + 1
                        numeric_inputs[col] = st.number_input(
                            field_label(col),
                            min_value=float(lo),
                            max_value=float(hi),
                            value=float((lo + hi) / 2),
                        )
                    else:
                        options = form_schema["categorical_options"][col]
                        categorical_inputs[col] = st.selectbox(field_label(col), options)

    analyze_clicked = st.form_submit_button("🔍 Analyze Claim", type="primary", use_container_width=True)

if analyze_clicked:
    if uploaded_image is None:
        st.error("Please upload a damage photo.")
    elif not description.strip():
        st.error("Please enter a claim description.")
    else:
        with st.spinner("Running CNN damage assessment..."):
            pil_img = Image.open(uploaded_image).convert("RGB").resize((224, 224))
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                np.expand_dims(np.array(pil_img).astype("float32"), axis=0)
            )
            heatmap, pred_idx = make_gradcam_heatmap(img_array, cnn_model)
            probs = cnn_model.predict(img_array, verbose=0)[0]
            severity = cnn_classes[pred_idx]
            confidence = float(probs[pred_idx])
            overlay_img = overlay_heatmap(pil_img, heatmap)

        with st.spinner("Running claim risk assessment..."):
            claim_fields = {**numeric_inputs, **categorical_inputs}
            row_df = pd.DataFrame([claim_fields])
            X_t = preprocessor.transform(row_df).astype("float32")
            risk_score = float(ann_model.predict(X_t, verbose=0).ravel()[0])

        with st.spinner("Gemini analyzing claim description..."):
            gemini_result = analyze_claim_text(
                description,
                {"severity": severity, "confidence": confidence},
                {"risk_score": risk_score},
            )

        st.session_state.analyzed = True
        st.session_state.result = {
            "severity": severity,
            "confidence": confidence,
            "overlay_img": overlay_img,
            "risk_score": risk_score,
            "gemini": gemini_result,
            "description": description,
            "claim_fields": claim_fields,
        }
        st.session_state.pop("decision_record", None)
        st.toast("Analysis complete — see the tabs below.", icon="✅")

if st.session_state.analyzed:
    result = st.session_state.result
    st.divider()
    st.subheader("2️⃣ Results")

    tab_findings, tab_decision, tab_report = st.tabs(
        ["🔍 AI Findings", "🧑‍⚖️ Human Decision", "📄 Report"]
    )

    with tab_findings:
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            with st.container(border=True):
                st.markdown("**Damage Assessment (CNN + Grad-CAM)**")
                st.image(
                    result["overlay_img"],
                    caption=f"Predicted: {result['severity']} ({result['confidence']:.0%} confidence)",
                    use_container_width=True,
                )

        with fcol2:
            with st.container(border=True):
                st.markdown("**Risk Assessment (ANN)**")
                st.metric("Estimated fraud/review risk", f"{result['risk_score']:.0%}")
                st.caption("Top contributing features (permutation importance)")
                imp_df = pd.DataFrame(
                    [{"feature": k.split("__", 1)[-1], "importance": v} for k, v in feature_importance.items()]
                ).sort_values("importance", ascending=True)
                st.bar_chart(imp_df.set_index("feature"))

        with st.container(border=True):
            st.markdown("**🧠 Gemini Reasoning Layer**")
            gemini = result["gemini"]
            flag_color = {
                "consistent": "green",
                "minor mismatch": "orange",
                "significant mismatch": "red",
            }.get(gemini["consistency_flag"], "gray")
            st.markdown(f"**Summary:** {gemini['summary']}")
            st.markdown(f"**Consistency check:** :{flag_color}[{gemini['consistency_flag']}]")
            st.markdown(f"**Recommendation for adjuster:** {gemini['recommendation']}")

    with tab_decision:
        st.markdown("**Adjuster decision**")
        decision = st.radio(
            "Decision", ["Approve", "Reject", "Modify"], horizontal=True, label_visibility="collapsed"
        )
        comment = st.text_area("Adjuster comment (optional for Approve, recommended for Reject/Modify)")

        if st.button("✅ Confirm Decision", type="primary"):
            st.session_state.decision_record = {
                "decision": decision,
                "comment": comment,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            log_row = pd.DataFrame(
                [
                    {
                        **st.session_state.decision_record,
                        "severity": result["severity"],
                        "risk_score": result["risk_score"],
                    }
                ]
            )
            log_row.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)
            st.success(f"Decision recorded: {decision}")

    with tab_report:
        if "decision_record" in st.session_state:
            pdf_bytes = build_report(result, st.session_state.decision_record)
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name="claim_report.pdf",
                mime="application/pdf",
                type="primary",
            )
        else:
            st.info("Confirm a decision in the 🧑‍⚖️ Human Decision tab first to generate the report.")
