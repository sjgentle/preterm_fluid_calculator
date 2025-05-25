# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model (must be uploaded to the same repo as this app)
model = joblib.load("fluid_risk_model.pkl")

# App title and description
st.title("Preterm Fluid Optimization Calculator")
st.markdown("This tool recommends a daily fluid volume (ml/kg) to reduce risk of BPD/Death in preterm infants, based on weight, urine output, sodium, and maturity.")

# User inputs
birthweight = st.number_input("Birthweight (g)", min_value=400, max_value=1500, value=800)
gest_age = st.number_input("Gestational Age (weeks)", min_value=22.0, max_value=30.0, value=26.0)
weight = st.number_input("Current Weight (g)", min_value=400, max_value=2000, value=800)
urine = st.number_input("Urine Output (ml/kg/hr)", min_value=0.0, max_value=10.0, value=2.5)
sodium = st.number_input("Serum Sodium (mmol/L)", min_value=120.0, max_value=160.0, value=138.0)
day = st.number_input("Postnatal Day", min_value=1, max_value=14, value=3)

# When user clicks the button
if st.button("Calculate Optimal Fluid Volume"):

    # Range of fluid values to evaluate
    fluid_range = np.arange(50, 181, 5)

    # Build candidate profiles
    inputs = pd.DataFrame({
        "fluid": fluid_range,
        "weight": [weight]*len(fluid_range),
        "urine": [urine]*len(fluid_range),
        "sodium": [sodium]*len(fluid_range),
        "day": [day]*len(fluid_range),
        "gestational_age": [gest_age]*len(fluid_range),
        "birthweight": [birthweight]*len(fluid_range)
    })

    # Predict risk for each fluid level
    probs = model.predict_proba(inputs)[:, 1]
    best_idx = np.argmin(probs)
    best_fluid = fluid_range[best_idx]
    min_risk = probs[best_idx]

    # Output result
    st.success(f"✅ Recommended Fluid: {best_fluid} ml/kg")
    st.write(f"🩺 Predicted BPD/Death Risk: {min_risk:.3f}")

    # Show chart
    st.line_chart(pd.DataFrame({
        "Fluid (ml/kg)": fluid_range,
        "Predicted Risk": probs
    }).set_index("Fluid (ml/kg)"))

