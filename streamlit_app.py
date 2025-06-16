import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model
xgb_model = joblib.load('models/xgb_model.pkl')

# Load background data for SHAP
background_df = pd.read_csv('data/insurance_cleaned.csv')
background_df = background_df.drop(columns=['charges'])
background_sample = background_df.sample(100, random_state=42)  

# Get the average of medical charges for comparison
average_charge = pd.read_csv('data/insurance_cleaned.csv')['charges'].mean()

# Application setup
st.title("Healthcare Cost Predictor")
st.markdown("Enter patient details below to predict their expected medical charges.")

# User input
age = st.slider('Age', 18, 100, 30)
bmi = st.slider('BMI', 10.0, 50.0, 25.0)
sex = st.selectbox('Sex', ['male', 'female'])
children = st.slider('Number of Children', 0, 5, 0)
smoker = st.selectbox('Smoker', ['yes', 'no'])
region = st.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])

# Convert input to model-ready format
input_data = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'children': [children],
    'sex_male': [1 if sex == 'male' else 0],
    'region_northwest': [1 if region == 'northwest' else 0],
    'region_southeast': [1 if region == 'southeast' else 0],
    'region_southwest': [1 if region == 'southwest' else 0],
    'smoker_yes': [1 if smoker == 'yes' else 0]
})

# Predict
if st.button('Predict Healthcare Cost'):
    prediction = xgb_model.predict(input_data)[0]

    # Debugging: Print columns to ensure they match
    #print(background_df.columns)
    #print(input_data.columns)

    # SHAP explanation
    explainer = shap.Explainer(xgb_model, background_sample)
    shap_values = explainer(input_data)

    # Display results
    st.success(f"Estimated Medical Charges: ${prediction:,.2f}")

    diff = prediction - average_charge
    percent_diff = (diff / average_charge) * 100

    if diff > 0:
        st.info(f"Your predicted charge is **{percent_diff:.1f}% higher** than the average (${average_charge:,.2f}).")
    elif diff < 0:
        st.info(f"Your predicted charge is **{abs(percent_diff):.1f}% lower** than the average (${average_charge:,.2f}).")
    else:
        st.info("Your predicted charge is exactly the same as the average.")
    
    st.caption("ℹ️ Estimated annual insurance charge based on patient risk profile.")

    # SHAP explanation section
    st.subheader("Understanding Your Cost Factors:")
    st.markdown("""
    **What does this chart mean?**

    The model starts with a base cost (around $14,226), then adjusts it based on your inputs.

    - 🔵 Blue bars **lower** your predicted cost
    - 🔴 Red bars **raise** it
    - The farther a bar stretches, the **bigger its impact**
    """)
    
    # SHAP waterfall plot
    fig, ax = plt.subplots(figsize=(10, 10))
    shap.plots.waterfall(shap_values[0], max_display=8, show=False)
    st.pyplot(fig)
    plt.clf()

    st.markdown("**Note:** This is a predictive model and should not replace professional medical advice.")
    st.markdown("Developed by James Petkin.")
    

