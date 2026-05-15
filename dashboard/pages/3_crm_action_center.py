import streamlit as st
import pandas as pd
import plotly.express as px
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from navigation import render_top_menu

# (Keep your existing st.set_page_config line here)
st.set_page_config(page_title="Customer Hub", page_icon="👥", layout="wide")

# ADD THIS LINE RIGHT HERE:
render_top_menu("CRM Action Center")

# ---------------------------------------------------------
# CRM ACTION CENTER MODULE
# Visualizes XGBoost Churn Predictions & SHAP Explanations
# ---------------------------------------------------------

st.set_page_config(page_title="CRM Action Center", page_icon="🎯", layout="wide")

@st.cache_data
def load_churn_data():
    file_path = "data_pipelines/features/churn_scores.parquet"
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    else:
        return None

st.title("🎯 CRM Action Center: Churn Prevention")
st.markdown("Use AI-driven predictions to identify high-risk customers and deploy targeted retention campaigns.")

df = load_churn_data()

if df is not None:
    # --- DATA PREPARATION ---
    # Convert churn risk from a decimal (0.85) to a clean percentage (85.0%)
    df['churn_risk_percentage'] = (df['churn_risk_score'] * 100).round(1)
    
    # Categorize customers into Action Tiers
    def assign_tier(score):
        if score >= 0.80: return "🔴 Critical Risk (Immediate Action)"
        elif score >= 0.50: return "🟡 Moderate Risk (Monitor)"
        else: return "🟢 Safe (Retained)"
        
    df['Risk Tier'] = df['churn_risk_score'].apply(assign_tier)
    
    # --- TOP ROW: KPI METRICS ---
    st.markdown("### 30-Day Retention Outlook")
    col1, col2, col3 = st.columns(3)
    
    critical_count = len(df[df['churn_risk_score'] >= 0.80])
    revenue_at_risk = df[df['churn_risk_score'] >= 0.80]['monetary'].sum()
    top_churn_reason = df['primary_churn_driver'].mode()[0].capitalize()
    
    col1.metric("Critical Risk Customers", f"{critical_count:,}")
    col2.metric("Revenue at Risk", f"${revenue_at_risk:,.2f}")
    col3.metric("Primary Churn Driver (SHAP)", top_churn_reason)
    
    st.divider()

    # --- MAIN VISUALIZATION: RISK DISTRIBUTION ---
    st.markdown("### Customer Risk Distribution")
    
    # Create a nice histogram showing how many customers fall into each risk bucket
    fig = px.histogram(
        df, 
        x="churn_risk_percentage", 
        color="Risk Tier",
        nbins=40,
        title="AI Churn Probability Distribution",
        labels={"churn_risk_percentage": "Probability of Churning (%)", "count": "Number of Customers"},
        color_discrete_map={
            "🔴 Critical Risk (Immediate Action)": "#ef4444",
            "🟡 Moderate Risk (Monitor)": "#f59e0b",
            "🟢 Safe (Retained)": "#10b981"
        }
    )
    fig.update_layout(bargap=0.1, height=400, margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)
    
    # --- TARGETED CAMPAIGN EXPORT ---
    st.markdown("### Targeted Retention Hit-List")
    st.markdown("Filter by top SHAP drivers to send personalized 'We Miss You' emails.")
    
    # Let the user filter the table
    selected_reason = st.selectbox("Filter by Primary Churn Reason:", ["All"] + list(df['primary_churn_driver'].unique()))
    
    display_df = df[df['churn_risk_score'] >= 0.80].sort_values(by='churn_risk_score', ascending=False)
    if selected_reason != "All":
        display_df = display_df[display_df['primary_churn_driver'] == selected_reason]
    
    # Clean up the display columns for the Marketing team
    display_cols = ['customer_id', 'Risk Tier', 'churn_risk_percentage', 'primary_churn_driver', 'recency', 'monetary']
    st.dataframe(
        display_df[display_cols].style.format({'churn_risk_percentage': '{:.1f}%', 'monetary': '${:.2f}'}), 
        use_container_width=True
    )

else:
    st.error("🚨 Churn data not found! Please run the Churn Engine first.")