# 🚀 Credit Card Fraud Detection System

An end-to-end Machine Learning system that detects fraudulent credit card transactions and serves predictions via a deployed FastAPI backend.

🔗 **Live API:**https://fraud-detection-api-vjah.onrender.com/docs

---

## 📌 Overview

This project focuses on detecting fraudulent financial transactions using machine learning on a highly imbalanced dataset. The system is built as a complete pipeline—from model training to deployment—with a working API and frontend interface.

---

## 🧠 Key Highlights

* Built a fraud detection model using **Logistic Regression**
* Handled **imbalanced dataset problem** using sampling techniques
* Evaluated using **Precision, Recall, and ROC-AUC**
* Developed a **FastAPI backend** for real-time inference
* Integrated a **frontend interface** for user interaction
* Implemented **threshold-based prediction control**
* Successfully deployed the API on cloud (Render)

---

## ⚙️ Tech Stack

* **Language:** Python
* **ML:** Scikit-learn
* **Backend:** FastAPI
* **Frontend:** HTML, JavaScript
* **Deployment:** Render

---

## 📊 Model Details

* Algorithm: Logistic Regression
* Dataset: Credit Card Fraud Detection Dataset
* Challenge: Extreme class imbalance
* Focus: Minimizing false negatives (fraud cases missed)

---

## 🧩 System Architecture

```text
User Input (Frontend / API)
        ↓
FastAPI Backend
        ↓
ML Model (model.pkl)
        ↓
Prediction + Probability Output
```

---

## 🚀 API Usage

### Endpoint

```http
POST /predict
```

---

### Sample Request

```json
{
  "V1": -10.64,
  "V2": 5.91,
  "V3": -11.67,
  ...
  "V28": -0.15,
  "Amount": 0.0,
  "Time": 41233
}
```

---

### Sample Response

```json
{
  "prediction": 1,
  "fraud_probability": 0.9836
}
```

---

## ▶️ Run Locally

### 1. Clone repo

```bash
git clone https://github.com/pandeypranjal/credit-card-fraud-detection-system.git
cd credit-card-fraud-detection-system
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run backend

```bash
uvicorn src.app:app --reload
```

---

### 4. Run frontend

```bash
cd frontend
python -m http.server 5500
```

---

### 5. Open in browser

```text
http://127.0.0.1:5500
```

---

## ⚠️ Dataset

Dataset is not included due to size limitations.

Download from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🧠 Key Learnings

* Real-world ML requires handling **imbalanced data**
* Accuracy alone is misleading → **Precision/Recall matter more**
* Deployment introduces challenges like:

  * Model loading
  * Path handling
  * API validation
* End-to-end systems are more valuable than standalone models

---

## 🔮 Future Improvements

* Add **Random Forest / XGBoost comparison**
* Implement **probability calibration**
* Improve frontend UX
* Add logging & monitoring
* Convert to real-time streaming system

---

## 👨‍💻 Author

**PandeyPranjal**
🔗 https://github.com/pandeypranjal
