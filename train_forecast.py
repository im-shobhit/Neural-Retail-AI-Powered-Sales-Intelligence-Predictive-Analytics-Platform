import pandas as pd
from prophet import Prophet

def train_baseline_model():
    print("Loading cleaned data...")
    df = pd.read_csv("data/processed/cleaned_retail.csv")
    
    # 1. Convert back to datetime (CSV saves dates as text!)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # 2. Extract just the Date (ignore the exact hour/minute)
    df['Date'] = df['InvoiceDate'].dt.date
    
    # 3. Group by Date to get total daily sales quantity
    print("Aggregating daily sales...")
    daily_sales = df.groupby('Date')['Quantity'].sum().reset_index()
    
    # 4. Rename columns to strictly match Prophet's requirements
    daily_sales.columns = ['ds', 'y']
    
    # 5. Initialize and train the AI
    print("Training Prophet model... (This might take a moment)")
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(daily_sales)
    
    # 6. Ask it to predict 30 days into the future
    print("Predicting the next 30 days...")
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # 7. Save the predictions
    output_path = "data/processed/sales_forecast.csv"
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(output_path, index=False)
    
    print(f"\n✅ Forecast successfully saved to: {output_path}")
    print("\nHere is a sneak peek at the last 5 days of predictions:")
    print(forecast[['ds', 'yhat']].tail())

if __name__ == "__main__":
    train_baseline_model()