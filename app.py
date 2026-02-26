from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from flask import Flask, jsonify, render_template, request

import sklearn.compose._column_transformer as ct


class _RemainderColsList(list):
    pass


ct._RemainderColsList = _RemainderColsList
# -----------------------------------------------------

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "deploy_model.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "deploy_features.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")

app = Flask(__name__)

print("Runtime versions:", {"sklearn": sklearn.__version__, "joblib": joblib.__version__})

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    DEPLOY_FEATURES: list[str] = json.load(f)

NUMERIC_FEATURES = [c for c in DEPLOY_FEATURES if c != "gender"]


def payload_to_df(payload: dict) -> pd.DataFrame:
    missing = [c for c in DEPLOY_FEATURES if c not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    row = {k: payload[k] for k in DEPLOY_FEATURES}

    if isinstance(row.get("gender"), str):
        row["gender"] = row["gender"].strip()

    for c in NUMERIC_FEATURES:
        row[c] = pd.to_numeric(row[c], errors="raise")

    return pd.DataFrame([row], columns=DEPLOY_FEATURES)


def _render_form(values=None, result=None, error=None):
    return render_template(
        "index.html",
        values=values or {},
        result=result,
        error=error,
        required_features=DEPLOY_FEATURES,
    )


@app.get("/")
def index():
    return _render_form()


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/info")
def info():
    return jsonify(
        message="Gaming Addiction Risk API",
        endpoints={"health": "/health", "predict": "/predict", "info": "/info"},
        required_features=DEPLOY_FEATURES,
    )


@app.post("/predict")
def predict():
    """
    - If JSON is sent -> return JSON (API)
    - If HTML form is submitted -> return HTML page with result (UI)
    """
    try:
        if request.is_json:
            payload = request.get_json(force=True)
            if not isinstance(payload, dict):
                return jsonify(error="JSON body must be an object with feature keys."), 400

            X = payload_to_df(payload)
            proba = float(model.predict_proba(X)[0, 1])
            pred = int(proba >= 0.5)
            return jsonify(prediction=pred, probability=proba)

        values = dict(request.form)

        payload = {
            "age": values.get("age"),
            "gender": values.get("gender"),
            "daily_gaming_hours": values.get("daily_gaming_hours"),
            "weekly_sessions": values.get("weekly_sessions"),
            "years_gaming": values.get("years_gaming"),
            "competitive_rank": values.get("competitive_rank"),
            "online_friends": values.get("online_friends"),
            "microtransactions_spending": values.get("microtransactions_spending"),
            "screen_time_total": values.get("screen_time_total"),
            "sleep_hours": values.get("sleep_hours"),
            "stress_level": values.get("stress_level"),
            "depression_score": values.get("depression_score"),
        }

        X = payload_to_df(payload)
        proba = float(model.predict_proba(X)[0, 1])
        pred = int(proba >= 0.5)

        result = {"prediction": pred, "probability": proba}
        return _render_form(values=values, result=result)

    except ValueError as e:
        return _render_form(values=dict(request.form), error=str(e)), 400
    except Exception as e:
        if request.is_json:
            return jsonify(error=f"{type(e).__name__}: {e}"), 500
        return _render_form(values=dict(request.form), error=f"{type(e).__name__}: {e}"), 500