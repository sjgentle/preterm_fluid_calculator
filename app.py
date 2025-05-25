# app.py
import streamlit as st
import pandas as pd
import numpy as np

# Dummy model simulating risk behavior
class DummyModel:
    def predict_proba(self, X):
        return np.array([[1 - 0.001*(abs(x["fluid"] - 120)), 0.001*(abs(x["fluid"] - 120))] for _, x in X.iterrows()])

model = DummyModel()

st.title("Preterm Fluid Optimization Calculator")
st.markdown("Estimate the optimal fluid volume to reduce risk of BPD/Death.")

bw = st.number_input("Birthweight (g)", min_value=400, max_value=1500, value=800)
ga = st.number_input("Gestational Age (weeks)", min_value=22.0, max_value=30.0, value=26.0)
weight = st.number_input("Current Weight (g)", min_value=400, max_value=2000, value=800)
urine = st.number_input("Urine Output (ml/kg/hr)", min_value=0.0, max_value=10.0, value=2.0)
sodium = st.number_input("Serum Sodium (mmol/L)", min_value=120.0, max_value=160.0, value=138.0)
day = st.number_input("Postnatal Day", min_value=1, max_value=14, value=3)

if st.button("Calculate Optimal Fluids"):
    fluid_range = np.arange(50, 181, 5)
    inputs = pd.DataFrame({
        "fluid": fluid_range,
        "weight": [weight]*len(fluid_range),
        "urine": [urine]*len(fluid_range),
        "sodium": [sodium]*len(fluid_range),
        "day": [day]*len(fluid_range),
        "gestational_age": [ga]*len(fluid_range),
        "birthweight": [bw]*len(fluid_range)
    })
    probs = model.predict_proba(inputs)[:, 1]
    best_idx = np.argmin(probs)
    st.success(f"Recommended fluid: {fluid_range[best_idx]} ml/kg")
    st.write(f"Predicted risk at this level: {probs[best_idx]:.3f}")
    st.line_chart(pd.DataFrame({"Fluid (ml/kg)": fluid_range, "Predicted Risk": probs}))
