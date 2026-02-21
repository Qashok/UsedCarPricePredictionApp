import streamlit as st
import requests

brand_list = [
    "Maruti", "Hyundai", "Honda", "Toyota", "Tata", "Mahindra",
    "Ford", "Chevrolet", "Volkswagen", "Skoda", "Renault", "Nissan",
    "Kia", "MG", "Jeep", "Fiat", "Datsun", "Audi", "BMW", "Mercedes-Benz",
    "Volvo", "Jaguar", "Land Rover", "Mini", "Porsche", "Lexus",
    "Isuzu", "Mitsubishi", "Force", "Opel", "Daewoo", "Ashok"
]

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

st.title("🚗 Car Selling Price Predictor (UI)")

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

# 🚫 model selector removed (KNN only)

if st.button("Predict"):
    payload = {
        "model": "knn",   # ✅ always KNN
        "data": {
            "brand": brand,
            "fuel": fuel,
            "owner": owner,
            "km_driven": float(km_driven)
        }
    }

    try:
        res = requests.post(
            "http://127.0.0.1:5000/predict",
            json=payload,
            timeout=10
        )
        out = res.json()

        if res.status_code != 200:
            st.error(out.get("error", "Unknown error"))
        else:
            st.success(f"✅ Predicted Price: ₹ {out['prediction']:,.0f}")

    except Exception as e:
        st.error(f"❌ Could not connect to Flask API. Is it running?\n\n{e}")
