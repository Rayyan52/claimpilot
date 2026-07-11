"""
Trains the claim risk/fraud ANN on the Insurance Fraud Detection tabular
dataset, saves the model + preprocessor + feature importance + a form
schema (used by the Streamlit app to build the claim-entry form).

Run: python scripts/train_ann.py
"""
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "claims")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

NUMERIC_COLS = [
    "months_as_customer", "age", "policy_deductable", "policy_annual_premium",
    "umbrella_limit", "capital-gains", "capital-loss", "incident_hour_of_the_day",
    "number_of_vehicles_involved", "bodily_injuries", "witnesses",
    "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim",
]
CATEGORICAL_COLS = [
    "insured_sex", "incident_severity", "incident_type",
    "authorities_contacted", "property_damage", "police_report_available",
]
TARGET_COL = "fraud_reported"


def find_data_file():
    matches = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
    matches += glob.glob(os.path.join(DATA_DIR, "**", "*.xlsx"), recursive=True)
    if not matches:
        raise FileNotFoundError(f"No CSV or XLSX found under {DATA_DIR}")
    return matches[0]


def load_data():
    path = find_data_file()
    print(f"Loading {path}")
    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(path, sheet_name=0)
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.replace("?", "Unknown")
    df[TARGET_COL] = df[TARGET_COL].map({"Y": 1, "N": 0})

    numeric_cols = [c for c in NUMERIC_COLS if c in df.columns]
    categorical_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = df[numeric_cols + categorical_cols + [TARGET_COL]].dropna(subset=[TARGET_COL])
    return df, numeric_cols, categorical_cols


def build_ann(input_dim):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC", "accuracy"])
    return model


def permutation_importance(model, X_test, y_test, feature_names, n_repeats=5):
    baseline_auc = roc_auc_score(y_test, model.predict(X_test, verbose=0).ravel())
    importances = {}
    rng = np.random.default_rng(42)
    for i, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X_test.copy()
            rng.shuffle(X_shuffled[:, i])
            auc = roc_auc_score(y_test, model.predict(X_shuffled, verbose=0).ravel())
            drops.append(baseline_auc - auc)
        importances[name] = float(np.mean(drops))
    return baseline_auc, importances


def main():
    df, numeric_cols, categorical_cols = load_data()
    X = df[numeric_cols + categorical_cols]
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])
    X_train_t = preprocessor.fit_transform(X_train).astype("float32")
    X_test_t = preprocessor.transform(X_test).astype("float32")
    feature_names = preprocessor.get_feature_names_out().tolist()

    model = build_ann(X_train_t.shape[1])
    model.fit(X_train_t, y_train, validation_split=0.15, epochs=40, batch_size=16, verbose=2)

    baseline_auc, importances = permutation_importance(model, X_test_t, y_test, feature_names)
    print(f"Test AUC: {baseline_auc:.3f}")
    top_features = dict(sorted(importances.items(), key=lambda kv: -kv[1])[:10])
    print("Top features:", top_features)

    model.save(os.path.join(MODELS_DIR, "risk_ann.h5"))
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "risk_preprocessor.pkl"))
    with open(os.path.join(MODELS_DIR, "risk_feature_importance.json"), "w") as f:
        json.dump(top_features, f, indent=2)

    form_schema = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "categorical_options": {
            c: sorted(df[c].dropna().unique().tolist()) for c in categorical_cols
        },
        "numeric_ranges": {
            c: [float(df[c].min()), float(df[c].max())] for c in numeric_cols
        },
        "test_auc": baseline_auc,
    }
    with open(os.path.join(MODELS_DIR, "risk_form_schema.json"), "w") as f:
        json.dump(form_schema, f, indent=2)

    print("Saved model, preprocessor, feature importance, and form schema to models/")


if __name__ == "__main__":
    main()
