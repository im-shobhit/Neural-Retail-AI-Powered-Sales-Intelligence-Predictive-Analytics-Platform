import streamlit as st
import pandas as pd
import plotly.express as px
import requests

def main():
    st.set_page_config(page_title="NeuralRetail Intelligence", layout="wide")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Dashboard:", [
        "📈 Demand Forecast", 
        "👥 Customer Segmentation",
        "📊 Exploratory Data Analysis",
        "📦 Inventory Optimization" # NEW PAGE ADDED HERE
    ])
    
    if page == "📈 Demand Forecast":
        render_forecast_page()
    elif page == "👥 Customer Segmentation":
        render_segmentation_page()
    elif page == "📊 Exploratory Data Analysis":
        render_eda_page()
    elif page == "📦 Inventory Optimization":
        render_inventory_page()

# --- Page 1: Demand Forecast ---
def render_forecast_page():
    st.title("📈 Demand Intelligence (Live API Version)")
    st.markdown("Fetching live predictions directly from our FastAPI backend server.")
    api_url = "http://127.0.0.1:8000/forecast"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        fig = px.line(df, x='ds', y='yhat', title="30-Day Demand Forecast")
        st.plotly_chart(fig, use_container_width=True)
    except requests.exceptions.RequestException:
        st.error("Failed to connect to backend API. Is FastAPI running?")

# --- Page 2: RFM Customer Clustering ---
def render_segmentation_page():
    st.title("👥 AI Customer Segmentation")
    st.markdown("Interactive 3D behavioral clustering using K-Means Machine Learning.")
    try:
        df = pd.read_csv("data/processed/rfm_clusters.csv")
    except FileNotFoundError:
        st.error("RFM data not found! Run rfm_clustering.py first.")
        return

    fig = px.scatter_3d(
        df, x='Recency', y='Frequency', z='Monetary',
        color='Segment', opacity=0.7,
        title="3D Customer Clusters",
        color_discrete_map={"VIPs": "#FFD700", "Loyal/Active": "#00FF00", "New/Occasional": "#1E90FF", "At-Risk": "#FF0000"}
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)

# --- Page 3: Exploratory Data Analysis ---
def render_eda_page():
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Discovering hidden trends and business insights from our historical data.")
    try:
        df = pd.read_csv("data/processed/cleaned_retail.csv")
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        price_col = 'Price' if 'Price' in df.columns else 'UnitPrice'
        invoice_col = 'Invoice' if 'Invoice' in df.columns else 'InvoiceNo'
        df['TotalPrice'] = df['Quantity'] * df[price_col]
    except FileNotFoundError:
        st.error("Cleaned data not found! Make sure you have data/processed/cleaned_retail.csv.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Best-Selling Products")
        top_products = df.groupby('Description')['Quantity'].sum().nlargest(10).reset_index()
        fig1 = px.bar(top_products, x='Quantity', y='Description', orientation='h', color='Quantity', color_continuous_scale='Blues')
        fig1.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.subheader("Busiest Days of the Week")
        df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
        cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df.groupby('DayOfWeek')[invoice_col].nunique().reindex(cats).reset_index()
        fig2 = px.bar(day_counts, x='DayOfWeek', y=invoice_col, color=invoice_col, color_continuous_scale='Purples')
        st.plotly_chart(fig2, use_container_width=True)

# --- Page 4: Inventory Optimization (NEW) ---
def render_inventory_page():
    st.title("📦 Inventory Optimization Engine")
    st.markdown("Dynamic Restock Targets for our Top 10 High-Volume Products based on sales velocity and a 7-day lead time.")
    
    try:
        df = pd.read_csv("data/processed/inventory_targets.csv")
    except FileNotFoundError:
        st.error("Inventory data not found! Run inventory_optimization.py first.")
        return

    # Let's style the dataframe to make the Reorder Point stand out
    st.dataframe(
        df.style.highlight_max(subset=['Reorder Point'], color='#ffcccc')
                .format({'Avg Daily Sales': "{:.1f}"}),
        use_container_width=True
    )
    
    st.info("💡 **How to read this table:** When the physical warehouse stock for a product drops below the **Reorder Point**, immediately call the supplier to order more. The **Safety Stock** provides a mathematical buffer to ensure we don't run out during the 7 days it takes the supplier to deliver.")

if __name__ == "__main__":
    main()