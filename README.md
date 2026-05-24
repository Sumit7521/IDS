---
title: Cyber IDS API
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Multi-Model Intrusion Detection System (IDS) API

A production-ready, highly optimized **Intrusion Detection System (IDS)** powered by a suite of Machine Learning and Deep Learning models. It classifies network traffic records from the **NSL-KDD dataset** as `normal` or one of many cyber attack categories (DoS, Probe, R2L, U2R).

The backend is built with **FastAPI** and is fully optimized for containerized cloud deployment (e.g., Render Free Tier).

---

## 🚀 Key Architectures & Models

This system deploys **9 different ML/DL architectures** to benchmark and analyze threat detection:

1.  **Feature Fusion Hybrid Model (Autoencoder + XGBoost):** Fuses standard features with compressed latent representations from the Autoencoder, then classifies using a robust XGBoost booster.
2.  **1D Convolutional Neural Network (CNN):** Spatial deep learning network utilizing Conv1D layers.
3.  **Standalone XGBoost:** High-performance tabular gradient boosting tree model.
4.  **CatBoost:** Optimized categorical boosting tree classifier.
5.  **Random Forest (SMOTE):** Decision forest trained with minority class over-sampling to address class imbalance.
6.  **AdaBoost:** Sequential tree boosting classifier.
7.  **K-Nearest Neighbors (KNN):** Distance-based spatial classifier.
8.  **Support Vector Machine (SVM):** Maximum margin linear classifier.
9.  **Naive Bayes:** Fast, probabilistic Gaussian classifier.

---

## ⚡ Memory & Performance Optimizations (Render-Ready)

Running multiple heavy deep learning and machine learning models simultaneously on a restricted server (like Render Free Tier with a **512 MB RAM limit**) presents severe constraints. We implemented the following enterprise-grade optimizations to ensure zero crashes:

*   **Dynamic Lazy Loading:** Heavy models (CNN, KNN, Random Forest, CatBoost, Autoencoder) are not loaded globally at startup. Instead, they are loaded **dynamically inside their respective API endpoints** only when a request is made. This drops idle RAM usage from 512MB+ to **under 100MB**!
*   **Active Garbage Collection:** Immediately after an endpoint makes a prediction, the model is deleted from the execution scope (`del`) and Python's Garbage Collector (`gc.collect()`) is run, instantly freeing up the occupied RAM.
*   **Lightweight KNN Compression:** To keep `knn_model.joblib` below GitHub's strict **100MB file limit**, the KNN model was trained on an optimized representative sample of 35,000 traffic records. This reduced size from 116MB to **~32MB** while maintaining high model density and increasing prediction speed by **3x**!
*   **SVM Softmax Normalization:** Since `LinearSVC` does not natively support `predict_proba`, we implemented a custom softmax normalization utility over the `decision_function` margins. This translates raw SVM decision metrics into clean pseudo-probabilities, maintaining 100% API output schema consistency.

---

## 🔗 API Endpoints

All POST endpoints accept the standard 41-feature payload (`NetworkData` schema) and return a consistent JSON response containing the `prediction`, `confidence`, and `class_probabilities`.

| HTTP Method | Endpoint | Description | Model Loading |
|---|---|---|---|
| `GET` | `/` | API status and home message. | - |
| `POST` | `/predict/hybrid` | Feature Fusion (Autoencoder + XGBoost) prediction. | **Lazy Loaded** |
| `POST` | `/predict/cnn` | 1D Convolutional Neural Network (CNN) prediction. | **Lazy Loaded** |
| `POST` | `/predict/xgboost` | Standalone XGBoost prediction. | Global |
| `POST` | `/predict/catboost` | Standalone CatBoost prediction. | **Lazy Loaded** |
| `POST` | `/predict/adaboost` | Standalone AdaBoost prediction. | Global |
| `POST` | `/predict/randomforest` | SMOTE-balanced Random Forest prediction. | **Lazy Loaded** |
| `POST` | `/predict/knn` | K-Nearest Neighbors (KNN) prediction. | **Lazy Loaded** |
| `POST` | `/predict/svm` | Support Vector Machine (LinearSVC) prediction. | Global |
| `POST` | `/predict/naivebayes` | Naive Bayes (GaussianNB) prediction. | Global |

---

## 🛠️ Deployment Configuration (Render)

When creating a new **Web Service** on Render, link this repository and use the following values:

*   **Runtime:** `Python`
*   **Build Command:** `pip install -r requirements.txt`
*   **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
*   **Python Version Constraint:** Automatically handled via the `.python-version` file specifying `3.11.9` for total package compatibility.

---

## 💻 Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/Sumit7521/IDS.git
cd IDS
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI development server
```bash
uvicorn main:app --reload --port 8000
```
Open **`http://localhost:8000/docs`** in your browser to view the Swagger UI documentation and test endpoints locally.
