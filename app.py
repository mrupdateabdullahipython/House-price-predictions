import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("🏠 House Price Prediction App")

# Inputs
area = st.number_input("Area", min_value=5000, max_value=12000)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=5)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=3)
garage = st.selectbox("Garage", [0,1])
year = st.number_input("Year Built", min_value=1995, max_value=2023)

location = st.selectbox("Location", ["Abuja","Kano","Kaduna","Jigawa","Lagos"])

# Encode location (IMPORTANT)
loc_Abuja = 1 if location == "Abuja" else 0
loc_Kano = 1 if location == "Kano" else 0
loc_Kaduna = 1 if location == "Kaduna" else 0
loc_Jigawa = 1 if location == "Jigawa" else 0

# Prediction
if st.button("Predict Price"):
    features = np.array([[area, bedrooms, bathrooms, garage, year,
                          loc_Abuja, loc_Kano, loc_Kaduna, loc_Jigawa]])

    prediction = model.predict(features)

    st.success(f"Estimated Price: ₦{int(prediction[0]):,}")