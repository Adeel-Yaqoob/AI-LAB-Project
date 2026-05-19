# 💳 Credit Card Fraud Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit">
  <img src="https://img.shields.io/badge/Scikit--Learn-Random%20Forest-orange?logo=scikitlearn">
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-purple?logo=pandas">
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-green">
</p>

---

## 📌 Overview

The Credit Card Fraud Detection System is a transaction monitoring dashboard designed to identify suspicious credit card transactions using machine learning classification techniques.

The application provides:
- Real-time fraud prediction
- Batch CSV transaction screening
- Dynamic model evaluation
- Interactive visualization dashboard
- Fraud probability analysis

The system is developed using Python and deployed through a Streamlit-based web interface.

---

# 🖥️ Dashboard Features

## 🔍 Real-Time Transaction Prediction
Users can:
- Select real fraud transactions
- Select legitimate transactions
- Enter custom transaction values
- Receive instant fraud probability prediction

---

## 📂 Batch Fraud Screening
Users can upload CSV files containing multiple transactions.

The system:
- Processes uploaded records
- Predicts fraud probability
- Labels transactions as Fraud or Legitimate
- Displays prediction summary charts

---

## 📊 Dynamic Model Evaluation
The dashboard dynamically computes:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

These metrics are calculated using the saved trained model and dataset test split.

---

# 🧠 Model Information

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/7/76/Random_forest_diagram_complete.png" width="500">
</p>

## Random Forest Classifier

The system uses the Random Forest Classification algorithm.

### Why Random Forest?
- High classification accuracy
- Strong performance on tabular datasets
- Handles fraud pattern complexity effectively
- Reduces overfitting through ensemble learning

The model predicts whether a transaction is:
- Legitimate
- Fraudulent

based on transaction patterns and feature relationships.

---

# 📁 Dataset Information

Dataset Source:

🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Dataset Characteristics
- Financial transaction dataset
- PCA-transformed features
- Binary classification problem
- Highly imbalanced fraud distribution

### Important Columns

| Feature | Description |
|---|---|
| Time | Time elapsed between transactions |
| Amount | Transaction amount |
| V1–V28 | PCA-transformed anonymized features |
| Class | Target variable (0 = Legitimate, 1 = Fraud) |

---

# ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend development |
| Streamlit | Web dashboard |
| Scikit-learn | Model training |
| Pandas | Data handling |
| Matplotlib | Data visualization |
| Joblib | Model serialization |

---

# 📊 Evaluation Metrics

The system evaluates model performance using:

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- Confusion Matrix

These metrics help analyze fraud detection reliability and classification effectiveness.

---

# 🏗️ Project Structure

```bash
Credit-Card-Fraud-Detection-System/
│
├── dataset/
│   └── sample_creditcard_10000.csv
│
├── models/
│   ├── fraud_model.pkl
│   └── scaler.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
````

---

# ▶️ Running The Application

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train Model

```bash
python train_model.py
```

---

## Run Dashboard

```bash
streamlit run app.py
```

---

# 📈 System Workflow

```text
Transaction Input
        ↓
Preprocessing & Scaling
        ↓
Random Forest Model
        ↓
Fraud Probability Prediction
        ↓
Dashboard Visualization
```

---

# 📷 Application Modules

## Prediction Module

Real-time single transaction prediction interface.

## Batch Screening Module

CSV upload and multiple transaction analysis.

## Evaluation Module

Dynamic performance metric visualization dashboard.

---

# 🔐 Data Privacy

The dataset uses PCA-transformed anonymized features to protect sensitive customer and banking information while preserving fraud detection patterns.

---

# 📌 Future Improvements

Possible future enhancements include:

* Cloud database integration
* Real-time transaction streaming
* Advanced anomaly detection models
* Multi-model comparison
* API integration
* Enterprise fraud monitoring workflows

---

# 👨‍💻 Developed For

Artificial Intelligence Lab Final Project


