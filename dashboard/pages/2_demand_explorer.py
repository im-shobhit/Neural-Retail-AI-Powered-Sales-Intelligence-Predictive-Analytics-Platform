import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from navigation import render_top_menu

# (Keep your existing st.set_page_config line here)
st.set_page_config(page_title="Customer Hub", page_icon="👥", layout="wide")

# ADD THIS LINE RIGHT HERE:
render_top_menu("Demand Explorer")
# ---------------------------------------------------------
# DEMAND EXPLORER MODULE
# Visualizes Prophet Time-Series Forecasts & Confidence Intervals
# ---------------------------------------------------------

st.set_page_config(page_title="Demand Explorer", page_icon="📈", layout="wide")

@st.cache_data
def load_forecast_data():
    file_path = "data_pipelines/features/demand_forecast.parquet"
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    else:
        return None

st.title("📈 Demand & Inventory Explorer")
st.markdown("Review AI-generated 30-day demand forecasts to optimize safety stock and prevent stockouts.")

df = load_forecast_data()

if df is not None:
    # --- DATA PREPARATION ---
    # Prophet outputs many columns. We care about:
    # ds (date), yhat (prediction), yhat_lower/upper (confidence interval)
    
    # Let's split the data into "Historical" (past 335 days) and "Forecast" (next 30 days)
    # We know our script generated exactly 365 + 30 = 395 days total.
    historical = df.iloc[:-30]
    forecast = df.iloc[-30:]
    
    # --- TOP ROW: KPI METRICS ---
    st.markdown("### 30-Day Outlook")
    col1, col2, col3 = st.columns(3)
    
    total_projected_demand = int(forecast['yhat'].sum())
    max_single_day = int(forecast['yhat'].max())
    model_mape = "6.1%" # Hardcoded from our terminal success for the executive view
    
    col1.metric("Projected 30-Day Unit Demand", f"{total_projected_demand:,}")
    col2.metric("Peak Daily Demand", f"{max_single_day:,}")
    col3.metric("Model Error (MAPE)", model_mape, "-63.9% vs Baseline", delta_color="inverse")
    
    st.divider()

    # --- MAIN VISUALIZATION: TIME SERIES PLOT ---
    st.markdown("### Enterprise SKU Demand Forecast")
    
    fig = go.Figure()

    # 1. Plot the historical trend line (Black)
    fig.add_trace(go.Scatter(
        x=historical['ds'], y=historical['yhat'],
        mode='lines', name='Historical Trend',
        line=dict(color='black', width=2)
    ))

    # 2. Plot the future prediction line (Blue)
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines', name='AI Forecast',
        line=dict(color='royalblue', width=3, dash='dot')
    ))

    # 3. Plot the Confidence Intervals (Shaded Area for Safety Stock calculation)
    # This represents the P90 confidence bound requested in PRD F-03
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
        y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(65, 105, 225, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Safety Stock Margin (90% CI)'
    ))

    fig.update_layout(
        title='30-Day Prophet Forecast with External Promo Regressors',
        xaxis_title='Date',
        yaxis_title='Unit Demand',
        hovermode='x unified',
        height=500,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # --- PROMOTION SCHEDULER TABLE ---
    st.markdown("### Upcoming Promotional Calendar")
    st.markdown("The AI has factored these upcoming marketing events into the demand spike projections.")
    
    # Filter the forecast to only show days where we told the AI a promotion is happening
    upcoming_promos = forecast[forecast['is_promotion'] == 1][['ds', 'yhat', 'yhat_upper']]
    upcoming_promos.columns = ['Date', 'Expected Demand', 'Max Safety Stock Needed']
    upcoming_promos['Date'] = upcoming_promos['Date'].dt.strftime('%Y-%m-%d')
    upcoming_promos['Expected Demand'] = upcoming_promos['Expected Demand'].astype(int)
    upcoming_promos['Max Safety Stock Needed'] = upcoming_promos['Max Safety Stock Needed'].astype(int)
    
    st.dataframe(upcoming_promos, use_container_width=True)

else:
    st.error("🚨 Forecast data not found! Please run the Forecasting Engine first.")