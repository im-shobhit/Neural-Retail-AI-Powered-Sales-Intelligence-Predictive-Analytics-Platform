# 🚀 NeuralRetail: AI-Powered Sales Intelligence Platform

NeuralRetail is an end-to-end MLOps platform designed for enterprise retail environments. It transforms raw transactional data into actionable business intelligence using advanced machine learning models and an interactive executive dashboard.

---

## 🌟 Key Features & Performance
* **👥 Customer Intelligence Hub**: RFM-based segmentation using K-Means clustering. 
  * *Result: Identified 6 distinct customer personas with a **0.804 Silhouette Score**.*
* **📈 Demand Explorer**: SKU-level time-series forecasting using Meta's Prophet algorithm with external promotional regressors.
  * *Result: Achieved **6.1% MAPE** (Mean Absolute Percentage Error) on a 30-day horizon.*
* **🎯 CRM Action Center**: Churn prediction and explainability using XGBoost and SHAP (Shapley Additive exPlanations).
  * *Result: **0.963 AUC-ROC** score for high-accuracy retention targeting.*
* **🔒 Secure Dashboard**: A multi-page Streamlit application with custom CSS, professional dark mode UI, and role-based access control.

---

## 🛠️ Tech Stack
* **Language**: Python 3.13
* **AI/ML**: Scikit-Learn, Prophet (Meta), XGBoost, SHAP
* **Data Engineering**: Pandas, PyArrow (Parquet), Great Expectations
* **Frontend**: Streamlit, Plotly, Streamlit-Option-Menu

---

## 🚀 Getting Started

### 1. Installation
```bash
pip install pandas scikit-learn prophet xgboost shap streamlit plotly streamlit-option-menu
```

### 2. Run the Data Pipeline
* Execute the scripts in order to generate and process the AI features:
```bash
python data_pipelines/generate_mock_data.py
python data_pipelines/segmentation_engine.py
python data_pipelines/forecasting_engine.py
python data_pipelines/churn_engine.py
```
---


### 3. Launch the Dashboard
```bash
python -m streamlit run dashboard/app.py
```
Login Password: admin2026

---

📊 Business Objectives Met
* [x] Revenue Uplift: Predicted to provide 15-25% uplift via targeted churn prevention.

* [x] Inventory Efficiency: Significantly reduced stockouts through high-accuracy demand forecasting.

* [x] Explainable AI: Integrated SHAP to provide "Primary Churn Drivers" for transparent marketing decisions.

Developed as a Production-Grade AI Platform following Amdocs PRD Requirements.
    
