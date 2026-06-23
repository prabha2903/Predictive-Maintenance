import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

st.title("Predictive Maintenance System")

values = []

for i in range(1,22):
    values.append(
        st.number_input(f"Sensor {i}", value=0.0)
    )

if st.button("Predict"):

    data = pd.DataFrame([values])

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.error("Maintenance Required")
    else:
        st.success("Machine Healthy")