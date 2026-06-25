# Predictive Maintenance System

## 📌 Overview

This project predicts whether industrial machinery requires maintenance using Machine Learning.

The system analyzes operational settings and sensor readings from the NASA Turbofan Engine Dataset and predicts machine health before failure occurs.

---

## 🚀 Features

- Predict Maintenance Requirement
- Machine Health Score
- Risk Level Classification
- CSV Upload Support
- Manual Sensor Input
- Sensor Visualization Dashboard
- Streamlit Deployment

---

## 📊 Dataset

NASA Turbofan Engine Degradation Dataset (CMAPSS)

Dataset contains:

- 3 Operational Settings
- 21 Sensor Measurements
- Engine Cycles
- Remaining Useful Life (RUL)

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Random Forest Classifier
- Matplotlib
- Streamlit

---

## 🤖 Machine Learning Workflow

1. Load NASA Dataset
2. Calculate Remaining Useful Life (RUL)
3. Create Maintenance Target Variable
4. Train Random Forest Model
5. Evaluate Model Accuracy
6. Save Model using Joblib
7. Deploy using Streamlit

---

## 📈 Dashboard Features

### Manual Input

Users can enter:

- Operational Settings
- Sensor Values

and instantly get:

- Health Score
- Risk Level
- Maintenance Prediction

### CSV Upload

Upload a CSV file containing machine sensor readings and receive maintenance predictions automatically.

---

## 🎯 Prediction Output

- ✅ Machine Healthy
- ⚠ Maintenance Required

Risk Levels:

- 🟢 Low Risk
- 🟡 Medium Risk
- 🔴 High Risk

---

## 🌐 Live Demo

https://predictive-maintenance-tdsrnkaocfftank9ycqkcu.streamlit.app/

---

## 📂 Project Structure

```
Predictive-Maintenance/
│
├── data/
├── models/
│   └── model.pkl
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

---

## 👨‍💻 Author

Prabha
B.E Computer Science and Engineering
Sri Shakthi Institute of Engineering and Technology
