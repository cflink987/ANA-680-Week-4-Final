from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

import sklearn.compose._column_transformer as ct

class _RemainderColsList(list):
    pass

ct._RemainderColsList = _RemainderColsList

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "deploy_model.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "deploy_features.json"

app = Flask(__name__)

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    DEPLOY_FEATURES = json.load(f)

NUMERIC_FEATURES = [c for c in DEPLOY_FEATURES if c != "gender"]

@app.get("/")
def home():
    return jsonify(
        message="Gaming Addiction Risk API",
        endpoints={"health": "/health", "predict": "/predict"},
        required_features=DEPLOY_FEATURES,
    )

@app.get("/health")
def health():
    return jsonify(status="ok")


def payload_to_df(payload: dict) -> pd.DataFrame:
    missing = [c for c in DEPLOY_FEATURES if c not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    row = {k: payload[k] for k in DEPLOY_FEATURES}

    for c in NUMERIC_FEATURES:
        row[c] = pd.to_numeric(row[c], errors="raise")

    if isinstance(row.get("gender"), str):
        row["gender"] = row["gender"].strip()

    return pd.DataFrame([row], columns=DEPLOY_FEATURES)


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            return jsonify(error="JSON body must be an object with feature keys."), 400

        X = payload_to_df(payload)

        proba = float(model.predict_proba(X)[0, 1])
        pred = int(proba >= 0.5)

        return jsonify(prediction=pred, probability=proba)

    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)