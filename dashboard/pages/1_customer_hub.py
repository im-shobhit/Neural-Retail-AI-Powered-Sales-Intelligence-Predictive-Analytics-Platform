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
render_top_menu("Customer Hub")

# ---------------------------------------------------------
# CUSTOMER HUB MODULE
# Visualizes the Output of our Segmentation Engine
# ---------------------------------------------------------

st.set_page_config(page_title="Customer Hub", page_icon="👥", layout="wide")

# MLOps Best Practice: Cache data loading so the dashboard doesn't re-read 
# the hard drive every time you click a button. (Keeps latency < 1.5s)
@st.cache_data
def load_segmentation_data():
    file_path = "data_pipelines/features/customer_segments.parquet"
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    else:
        return None

st.title("👥 Customer Intelligence Hub")
st.markdown("Explore AI-generated customer personas based on Recency, Frequency, and Monetary (RFM) behavioral patterns.")

df = load_segmentation_data()

if df is not None:
    # --- TOP ROW: KPI METRICS ---
    st.markdown("### Executive Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(df)
    total_revenue = df['monetary'].sum()
    avg_order_val = total_revenue / df['frequency'].sum()
    num_segments = df['cluster'].nunique()
    
    col1.metric("Total Active Customers", f"{total_customers:,}")
    col2.metric("Total Lifetime Value", f"${total_revenue:,.2f}")
    col3.metric("Avg Order Value", f"${avg_order_val:.2f}")
    col4.metric("AI Personas Identified", num_segments)
    
    st.divider()

    # --- MAIN VISUALIZATION: 3D CLUSTER PLOT ---
    st.markdown("### Behavioral Cluster Distribution (3D)")
    st.markdown("This interactive visualization shows how our K-Means algorithm separated customers into distinct behavioral islands.")
    
    # We must convert 'cluster' to a string so Plotly treats it as a category, not a continuous number
    df['Persona'] = "Segment " + df['cluster'].astype(str)
    
    fig = px.scatter_3d(
        df, 
        x='recency', 
        y='frequency', 
        z='monetary',
        color='Persona',
        hover_data=['customer_id'],
        title="RFM Cluster Matrix",
        labels={'recency': 'Days Since Last Order', 'frequency': 'Total Orders', 'monetary': 'Total Spend ($)'},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    # Make the chart look sleek and enterprise-ready
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40), height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # --- DATA TABLE EXPORT ---
    st.markdown("### Segment Export")
    st.markdown("Marketing teams can export these segments to CSV for targeted email campaigns.")
    st.dataframe(df.head(50), use_container_width=True)

else:
    st.error("🚨 Data not found! Please run the Segmentation Engine pipeline first to generate `customer_segments.parquet`.")