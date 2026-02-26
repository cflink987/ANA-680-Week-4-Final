from __future__ import annotations

import json
import os
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
BOUNDS_PATH = ARTIFACTS_DIR / "input_bounds.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")
if not BOUNDS_PATH.exists():
    raise FileNotFoundError(f"Missing bounds file: {BOUNDS_PATH}")

app = Flask(__name__)

print("Runtime versions:", {"sklearn": sklearn.__version__, "joblib": joblib.__version__})

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    DEPLOY_FEATURES: list[str] = json.load(f)

with open(BOUNDS_PATH, "r", encoding="utf-8") as f:
    INPUT_BOUNDS = json.load(f)

INPUT_POLICY = os.getenv("INPUT_POLICY", INPUT_BOUNDS.get("policy_default", "clip")).lower()

NUMERIC_FEATURES = [c for c in DEPLOY_FEATURES if c != "gender"]

LABELS = {
    "age": "age",
    "gender": "gender",
    "daily_gaming_hours": "daily gaming hours",
    "weekly_sessions": "weekly sessions",
    "years_gaming": "years gaming",
    "competitive_rank": "competitive rank",
    "online_friends": "online friends",
    "microtransactions_spending": "microtransactions spending",
    "screen_time_total": "screen time total",
    "sleep_hours": "sleep hours",
    "stress_level": "stress level",
    "depression_score": "depression score",
}


def validate_and_normalize(payload: dict) -> tuple[dict, dict]:
    """
    Returns (cleaned_payload, clipped_fields)
    clipped_fields: {feature: {"from": original, "to": clipped}}
    """
    feat_bounds = INPUT_BOUNDS["features"]
    allowed_gender = set(feat_bounds["gender"]["allowed"])

    cleaned: dict = {}
    clipped: dict = {}

    for f in DEPLOY_FEATURES:
        if f not in payload:
            raise ValueError(f"Missing required field: {f}")

        raw = payload[f]

        if f == "gender":
            g = str(raw).strip()
            if g not in allowed_gender:
                raise ValueError(f"gender must be one of {sorted(allowed_gender)}")
            cleaned[f] = g
            continue

        x = float(raw)
        lo = float(feat_bounds[f]["min"])
        hi = float(feat_bounds[f]["max"])

        if x < lo or x > hi:
            if INPUT_POLICY == "reject":
                raise ValueError(f"{f} out of range [{lo}, {hi}] (got {x})")
            x2 = min(max(x, lo), hi)
            clipped[f] = {"from": float(x), "to": float(x2)}
            x = x2

        cleaned[f] = x

    return cleaned, clipped


def payload_to_df(payload: dict) -> pd.DataFrame:
    # assumes payload already validated & normalized
    return pd.DataFrame([payload], columns=DEPLOY_FEATURES)


def render_home(values=None, result=None, error=None, clipped_fields=None):
    return render_template(
        "index.html",
        values=values or {},
        result=result,
        error=error,
        clipped_fields=clipped_fields or {},
        bounds=INPUT_BOUNDS,
        policy=INPUT_POLICY,
        labels=LABELS,
        required_features=DEPLOY_FEATURES,
    )


@app.get("/")
def home():
    # This is what "Open app" shows
    return render_home()


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/info")
def info():
    return jsonify(
        message="Gaming Addiction Risk API",
        endpoints={"health": "/health", "predict": "/predict", "info": "/info"},
        required_features=DEPLOY_FEATURES,
        input_policy=INPUT_POLICY,
        bounds_type=INPUT_BOUNDS.get("bounds_type"),
    )


@app.post("/predict")
def predict():
    """
    - If JSON sent -> JSON response
    - If HTML form -> returns HTML page with result
    """
    try:
        if request.is_json:
            payload = request.get_json(force=True)
            if not isinstance(payload, dict):
                return jsonify(error="JSON body must be an object with feature keys."), 400

            cleaned, clipped_fields = validate_and_normalize(payload)
            X = payload_to_df(cleaned)

            proba = float(model.predict_proba(X)[0, 1])
            pred = int(proba >= 0.5)

            return jsonify(
                prediction=pred,
                probability=proba,
                policy=INPUT_POLICY,
                bounds_type=INPUT_BOUNDS.get("bounds_type"),
                clipped_fields=clipped_fields,
            )

        form_vals = dict(request.form)

        payload = {f: form_vals.get(f) for f in DEPLOY_FEATURES}
        cleaned, clipped_fields = validate_and_normalize(payload)

        X = payload_to_df(cleaned)
        proba = float(model.predict_proba(X)[0, 1])
        pred = int(proba >= 0.5)

        values = {k: str(cleaned[k]) for k in DEPLOY_FEATURES}
        result = {"prediction": pred, "probability": proba}

        return render_home(values=values, result=result, clipped_fields=clipped_fields)

    except ValueError as e:
        if request.is_json:
            return jsonify(error=str(e)), 400
        return render_home(values=dict(request.form), error=str(e)), 400

    except Exception as e:
        if request.is_json:
            return jsonify(error=f"{type(e).__name__}: {e}"), 500
        return render_home(values=dict(request.form), error=f"{type(e).__name__}: {e}"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)