from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# -----------------------------
# Always load files from SAME folder as this script
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pkl(name: str):
    path = os.path.join(BASE_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return joblib.load(path)

# -----------------------------
# Load saved artifacts (ONLY these 4 files)
# -----------------------------
transform = load_pkl("transformer.pkl")     # ColumnTransformer
perms = load_pkl("features.pkl")            # selected feature names after transform
knn = load_pkl("knn_model.pkl")
lr  = load_pkl("liner_model.pkl")

# normalize perms to python list
if isinstance(perms, (np.ndarray, pd.Index)):
    perms = perms.tolist()
elif not isinstance(perms, list):
    perms = list(perms)

def preprocess_input(payload: dict) -> pd.DataFrame:
    """
    Accepts either:
    {brand,fuel,owner,km_driven, model}
    OR
    {model: "...", data: {brand,fuel,owner,km_driven}}
    """
    required = ["brand", "fuel", "owner", "km_driven"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    brand = str(payload["brand"])
    fuel  = str(payload["fuel"])
    owner = str(payload["owner"])

    try:
        km = float(payload["km_driven"])
    except Exception:
        raise ValueError("km_driven must be a number")

    # Single-row DF with EXACT columns used in training
    X = pd.DataFrame([{
        "brand": brand,
        "fuel": fuel,
        "owner": owner,
        "km_driven": km
    }])

    return X

def make_features(X_raw: pd.DataFrame) -> pd.DataFrame:
    """
    1) ColumnTransformer -> numpy array
    2) Convert to DataFrame with output feature names
    3) Select columns using perms (features.pkl)
    """
    # Ensure transformer input columns exist (safety)
    try:
        expected_in = list(getattr(transform, "feature_names_in_", []))
        for col in expected_in:
            if col not in X_raw.columns:
                X_raw[col] = 0.0
        # Keep same order if feature_names_in_ exists
        if expected_in:
            X_raw = X_raw[expected_in]
    except Exception:
        pass

    X_enc = transform.transform(X_raw)

    # Feature names after ColumnTransformer
    try:
        feat_names = transform.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(X_enc.shape[1])]

    X_df = pd.DataFrame(X_enc, columns=feat_names)

    # Ensure all expected columns from training exist
    for c in perms:
        if c not in X_df.columns:
            X_df[c] = 0.0

    # Keep only training-time selected features in the same order
    return X_df[perms]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True) or {}

        # allow both formats:
        # 1) {brand,fuel,owner,km_driven, model}
        # 2) {model: "...", data: {brand,fuel,owner,km_driven}}
        data = payload.get("data", payload)

        model_choice = (payload.get("model") or data.get("model") or "knn").lower()

        X_raw = preprocess_input(data)
        X_final = make_features(X_raw)

        if model_choice in ("lr", "linear", "linear_regression", "logical"):
            pred = lr.predict(X_final)[0]
            used = "linear_regression"
        else:
            pred = knn.predict(X_final)[0]
            used = "knn"

        return jsonify({"model": used, "prediction": float(pred)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict_form", methods=["GET"])
def predict_form():
    return jsonify({
        "how_to_use": "POST JSON to /predict with keys brand,fuel,owner,km_driven and optional model=knn|lr",
        "example_payload": {
            "brand": "Maruti",
            "fuel": "Diesel",
            "owner": "First Owner",
            "km_driven": 45000,
            "model": "knn"
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
