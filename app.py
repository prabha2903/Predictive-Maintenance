import matplotlib.pyplot as plt
import streamlit as st
import joblib
import pandas as pd

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# ==========================
# Custom CSS
# ==========================
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

div[data-testid="metric-container"] {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Load Model
# ==========================
model = joblib.load("models/model.pkl")

# ==========================
# Header
# ==========================
st.title("🔧 AI Predictive Maintenance Dashboard")

st.markdown(
    "Predict machine failures before they happen using Machine Learning."
)

# ==========================
# Sidebar
# ==========================
st.sidebar.title("📌 Project Info")

st.sidebar.success("✅ AI Model Loaded")

st.sidebar.info("Dataset: NASA Turbofan Engine Dataset")

st.sidebar.info("Model: Random Forest Classifier")

st.sidebar.markdown("---")

st.sidebar.metric("Features", "24")
st.sidebar.metric("Sensors", "21")

# ==========================
# CSV Upload Section
# ==========================
st.header("📂 Option 1: Upload CSV")

uploaded_file = st.file_uploader(
    "Upload Sensor CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    if st.button("Predict from CSV"):

        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0]

        health_score = probability[0] * 100

        st.markdown("## 📊 Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Health Score",
                f"{health_score:.2f}%"
            )

        with col2:
            st.metric(
                "Status",
                "Healthy" if prediction == 0 else "Maintenance"
            )

        with col3:
            st.metric(
                "Risk",
                "Low" if health_score > 80
                else "Medium" if health_score > 50
                else "High"
            )

        st.progress(int(health_score))

        st.caption(
            f"Machine operating at {health_score:.2f}% health."
        )

        # Sensor Visualization
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            range(len(data.columns)),
            data.iloc[0],
            marker="o"
        )

        ax.set_title("Uploaded Sensor Data")
        ax.set_xlabel("Features")
        ax.set_ylabel("Value")

        st.pyplot(fig)

        if prediction == 1:
            st.error("⚠ Maintenance Required")
        else:
            st.success("✅ Machine Healthy")

# ==========================
# Manual Input Section
# ==========================
st.header("⌨️ Option 2: Manual Input")

values = []

st.subheader("Operational Settings")

for i in range(1, 4):
    values.append(
        st.number_input(
            f"Setting {i}",
            value=0.0,
            key=f"setting_{i}"
        )
    )

st.subheader("Sensor Values")

col1, col2, col3 = st.columns(3)

for i in range(1, 8):
    with col1:
        values.append(
            st.number_input(
                f"Sensor {i}",
                value=0.0,
                key=f"sensor_{i}"
            )
        )

for i in range(8, 15):
    with col2:
        values.append(
            st.number_input(
                f"Sensor {i}",
                value=0.0,
                key=f"sensor_{i}"
            )
        )

for i in range(15, 22):
    with col3:
        values.append(
            st.number_input(
                f"Sensor {i}",
                value=0.0,
                key=f"sensor_{i}"
            )
        )

if st.button(
    "🚀 Predict Machine Health",
    use_container_width=True
):

    columns = (
        [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    data = pd.DataFrame(
        [values],
        columns=columns
    )

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0]

    health_score = probability[0] * 100

    st.markdown("## 📊 Prediction Result")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Health Score",
            f"{health_score:.2f}%"
        )

    with c2:
        st.metric(
            "Status",
            "Healthy" if prediction == 0 else "Maintenance"
        )

    with c3:
        st.metric(
            "Risk",
            "Low" if health_score > 80
            else "Medium" if health_score > 50
            else "High"
        )

    st.progress(int(health_score))

    st.caption(
        f"Machine operating at {health_score:.2f}% health."
    )

    # Sensor Graph
    sensor_values = values[3:]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        range(1, 22),
        sensor_values,
        marker="o"
    )

    ax.set_title("Sensor Readings")
    ax.set_xlabel("Sensor Number")
    ax.set_ylabel("Sensor Value")

    st.pyplot(fig)

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