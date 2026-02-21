import os
import joblib
import pandas as pd
import streamlit as st

brand_list = [
    "Maruti", "Hyundai", "Honda", "Toyota", "Tata", "Mahindra",
    "Ford", "Chevrolet", "Volkswagen", "Skoda", "Renault", "Nissan",
    "Kia", "MG", "Jeep", "Fiat", "Datsun", "Audi", "BMW", "Mercedes-Benz",
    "Volvo", "Jaguar", "Land Rover", "Mini", "Porsche", "Lexus",
    "Isuzu", "Mitsubishi", "Force", "Opel", "Daewoo", "Ashok"
]

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Car Selling Price Predictor")
st.caption("Deployed on Streamlit Cloud • Model: KNN")

# -----------------------------
# Load artifacts from same folder as this file
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    knn = joblib.load(os.path.join(BASE_DIR, "knn_model.pkl"))
    transformer = joblib.load(os.path.join(BASE_DIR, "transformer.pkl"))
    features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))
    return knn, transformer, features

knn, transformer, features = load_artifacts()

# normalize features to python list
if not isinstance(features, list):
    try:
        features = features.tolist()
    except Exception:
        features = list(features)

# -----------------------------
# UI Inputs
# -----------------------------
brand = st.selectbox("Brand", sorted(brand_list))
fuel = st.selectbox("Fuel", ["Diesel", "Petrol", "CNG", "LPG", "Electric"])
owner = st.selectbox(
    "Owner",
    ["Test Drive Car", "First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]
)
km_driven = st.number_input(
    "KM Driven",
    min_value=0,
    max_value=1_000_000,
    value=45000,
    step=1000
)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict"):
    try:
        X = pd.DataFrame([{
            "brand": brand,
            "fuel": fuel,
            "owner": owner,
            "km_driven": float(km_driven)
        }])

        # If "features.pkl" includes raw feature names used before transform,
        # align to that order. Otherwise, fallback to X columns.
        cols = [c for c in features if c in X.columns]
        if cols:
            X = X[cols]

        X_tr = transformer.transform(X)
        pred = float(knn.predict(X_tr)[0])

        st.success(f"✅ Predicted Price: ₹ {pred:,.0f}")

    except Exception as e:
        st.error("❌ Prediction failed. This usually happens if feature names/order don't match training.")
        st.exception(e)