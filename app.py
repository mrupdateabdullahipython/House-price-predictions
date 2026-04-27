import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title="House Price Predictor", layout="wide")

# 🔥 TITLE + DESCRIPTION
st.markdown("<h1 style='color:#2E8B57;'>🏠 House Price Prediction App</h1>", unsafe_allow_html=True)
st.write("This app predicts house prices based on area, bedrooms, bathrooms, garage, year built, and location using a Machine Learning model.")

# 📊 LOAD DATA (for charts)
df = pd.read_csv("created_house_price_prediction.csv")

# 🔲 LAYOUT (2 COLUMNS)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Input Features")

    area = st.number_input("Area", 5000, 12000)
    bedrooms = st.number_input("Bedrooms", 1, 5)
    bathrooms = st.number_input("Bathrooms", 1, 3)
    garage = st.selectbox("Garage", [0,1])
    year = st.number_input("Year Built", 1995, 2023)

    location = st.selectbox("Location", ["Abuja","Kano","Kaduna","Jigawa","Lagos"])

# 🔑 ENCODE LOCATION
loc_Abuja = 1 if location == "Abuja" else 0
loc_Kano = 1 if location == "Kano" else 0
loc_Kaduna = 1 if location == "Kaduna" else 0
loc_Jigawa = 1 if location == "Jigawa" else 0

# 🎯 PREDICTION
with col2:
    st.subheader("📈 Prediction")

    if st.button("Predict Price"):
        features = np.array([[area, bedrooms, bathrooms, garage, year,
                              loc_Abuja, loc_Kano, loc_Kaduna, loc_Jigawa]])

        prediction = model.predict(features)

        st.success(f"Estimated Price: ₦{int(prediction[0]):,}")

# 📊 CHARTS SECTION
st.markdown("---")
st.subheader("📊 Data Insights")

col3, col4 = st.columns(2)

# Scatter Plot
with col3:
    st.write("Area vs Price")
    fig, ax = plt.subplots()
    ax.scatter(df['Area'], df['Price'])
    ax.set_xlabel("Area")
    ax.set_ylabel("Price")
    st.pyplot(fig)

# Histogram
with col4:
    st.write("Price Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(df['Price'])
    st.pyplot(fig2)

# 🧠 MODEL EXPLANATION
st.markdown("👨‍💻Project Created by Built mrupdateabdullahi")
st.subheader("🧠 Model Explanation")

st.write("""
This model uses **Linear Regression**, which learns the relationship between input features and house prices.

- Area, Bedrooms, Bathrooms increase price 📈  
- Older houses reduce price 📉  
- Location significantly affects price 💰  

The model achieved a high accuracy (R² ≈ 0.9), meaning it can explain about 90% of the price variation.
""")

# 📌 FOOTER
st.markdown("👨‍💻 Project Created by mrupdateabdullahi")
st.caption("Built with ❤️ using Streamlit & Machine Learning")