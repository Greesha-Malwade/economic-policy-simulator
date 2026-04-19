# 💹 Economic Policy Simulator (MLOps + DevOps Project)

## 📌 Overview

This project is an **AI-powered Economic Policy Simulator** that predicts key macroeconomic indicators like **inflation** and **unemployment** based on government policies.

It demonstrates a complete **End-to-End ML + DevOps Pipeline**, including model training, API deployment, CI/CD automation, and visualization.

---

## 🎯 Problem Statement

Build and deploy an end-to-end machine learning pipeline that:

* Simulates economic scenarios
* Predicts inflation and unemployment
* Provides policy recommendations
* Automates workflows using DevOps practices

---

## ⚙️ Features

* 📊 Predicts **Inflation (%)** and **Unemployment (%)**
* 🧠 Uses **Machine Learning (Linear Regression)**
* 🌐 Interactive **Flask Web Interface**
* 🔁 Automated **CI/CD Pipeline (GitHub Actions)**
* 📈 Performance evaluation with metrics & graphs
* 💡 Rule-based **policy recommendations**

---

## 🏗️ Project Architecture

User Input (UI)
→ Flask API
→ ML Models (Inflation & Unemployment)
→ Predictions
→ Policy Recommendation

---

## 🧠 ML Models Used

* Linear Regression (Scikit-learn)
* Separate models for:

  * Inflation Prediction
  * Unemployment Prediction

---

## 📊 Inputs

* Interest Rate (%)
* Government Spending (Billion $)
* Tax Rate (%)

---

## 📈 Outputs

* Predicted Inflation (%)
* Predicted Unemployment (%)
* Policy Recommendation:

  * Expansionary Policy
  * Contractionary Policy
  * Neutral Policy

---

## 🚀 How to Run Locally

### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/economic-policy-simulator.git
cd economic-policy-simulator
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model

```bash
python src/train.py
```

### 4. Run Application

```bash
cd app
python app.py
```

### 5. Open in Browser

```
http://127.0.0.1:5000
```

---

## 🔁 CI/CD Pipeline

Implemented using **GitHub Actions**:

* Automatically installs dependencies
* Trains ML models
* Runs tests
* Builds Docker image

---

## 🐳 Docker (Optional)

```bash
docker build -t economic-policy .
docker run -p 5000:5000 economic-policy
```

---

## 📂 Project Structure

```
economic-policy-simulator/
│
├── app/                # Flask application
├── src/                # Model training
├── models/             # Saved ML models
├── data/               # Dataset
├── tests/              # Test cases
├── .github/workflows/  # CI/CD pipeline
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📊 Performance Metrics

* Inflation Model: R² ≈ 0.97
* Unemployment Model: R² ≈ 0.92

---

## 🛠️ Tech Stack

* Python
* Flask
* Scikit-learn
* Pandas, NumPy
* Matplotlib
* Git & GitHub
* GitHub Actions (CI/CD)
* Docker

---

## 📌 Future Enhancements

* Use real-world economic datasets
* Add advanced ML models (XGBoost, Neural Networks)
* Deploy on cloud (AWS/GCP)
* Add dashboards (Power BI / Streamlit)

---

## 👩‍💻 Author

**Greesha Malwade**

---

## ⭐ Conclusion

This project successfully demonstrates a **complete MLOps + DevOps pipeline**, integrating machine learning with real-world deployment and automation practices.
