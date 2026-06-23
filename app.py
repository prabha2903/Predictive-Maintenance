import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

st.title("Predictive Maintenance System")

values = []

st.subheader("Operational Settings")

for i in range(1, 4):
    values.append(
        st.number_input(f"Setting {i}", value=0.0)
    )

st.subheader("Sensor Values")

for i in range(1, 22):
    values.append(
        st.number_input(f"Sensor {i}", value=0.0)
    )
if st.button("Predict"):

    columns = (
        [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    data = pd.DataFrame([values], columns=columns)

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0]

    health_score = probability[0] * 100

    st.metric(
        label="Machine Health",
        value=f"{health_score:.2f}%"
    )

    if health_score > 80:
        st.success("🟢 Low Risk")
    elif health_score > 50:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

    if prediction == 1:
        st.error("⚠ Maintenance Required")
    else:
        st.success("✅ Machine Healthy")