# 🧠 NeuralRetail: AI-Powered Sales Intelligence & Predictive Analytics Platform

An end-to-end Machine Learning and Business Intelligence architecture designed to optimize B2B wholesale operations. This platform bridges the gap between raw transaction data and actionable business strategy by utilizing predictive modeling, behavioral clustering, and a decoupled microservice full-stack deployment.

<br>
<br>

## 🏗️ System Architecture

This project is built using a decoupled microservice architecture, separating the Machine Learning inference engine from the client-facing presentation layer.

* **Backend Engine:** FastAPI (Python) serving live predictive data.
* **Frontend UI:** Streamlit multi-page interactive dashboard.
* **Data & ML Stack:** Pandas, Scikit-Learn (Random Forest, K-Means), Prophet, Plotly.

<br>
<br>

## 🚀 Core Features & ML Models

### 1. Demand Forecasting (Time-Series)
* Utilizes **Prophet** to analyze historical sales data and forecast future 30-day demand volume.
* Deployed as a live REST API endpoint (`/forecast`) via FastAPI for real-time frontend consumption.

### 2. AI Customer Segmentation (Unsupervised Learning)
* Engineered an **RFM (Recency, Frequency, Monetary)** data matrix from raw transaction logs.
* Trained a **K-Means Clustering** algorithm to dynamically group customers into behavioral segments (VIPs, Loyal, At-Risk, New) based on multi-dimensional mathematical proximity.
* Visualized via an interactive 3D scatter plot.

### 3. Churn Prediction (Supervised Learning)
* Engineered advanced behavioral features including *Customer Tenure* and *Purchase Pace*.
* Trained a **Random Forest Classifier** to identify "At-Risk" customers with a high probability of churning, enabling targeted marketing interventions.

### 4. Inventory Optimization Engine
* Applied Supply Chain mathematics to calculate **Safety Stock** and **Reorder Points (ROP)** for top-selling products.
* Implemented algorithmic guardrails to handle `NaN` edge cases caused by single-day massive bulk orders.

<br>
<br>

## ⚙️ How to Run Locally

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/yourusername/NeuralRetail.git](https://github.com/yourusername/NeuralRetail.git)
cd NeuralRetail
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
pip install -r requirements.txt
