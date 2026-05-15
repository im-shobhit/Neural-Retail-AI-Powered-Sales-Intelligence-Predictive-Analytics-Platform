import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import os
from datetime import datetime, timedelta

# ---------------------------------------------------------
# DEMAND FORECASTING ENGINE (PRD Requirement F-03)
# Time-Series Forecasting + External Regressors (Promotions)
# ---------------------------------------------------------

def generate_predictable_demand():
    """Generates 365 days of realistic, predictable retail demand."""
    print("Generating predictable time-series data (Trend + Weekly Seasonality + Promos)...")
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=365)
    df = pd.DataFrame({'ds': dates})
    
    # 1. Base Trend (Sales naturally growing over the year)
    trend = np.linspace(50, 150, 365)
    
    # 2. Weekly Seasonality (Weekends sell 40% more)
    day_of_week = df['ds'].dt.dayofweek
    weekly_seasonality = np.where(day_of_week >= 5, 1.4, 1.0)
    
    # 3. External Regressor: Promotions (Random marketing spikes)
    df['is_promotion'] = np.random.choice([0, 1], size=365, p=[0.9, 0.1])
    promo_spike = np.where(df['is_promotion'] == 1, 1.5, 1.0) # 50% boost on promo days
    
    # 4. Combine and add a tiny bit of realistic noise
    noise = np.random.normal(0, 5, 365)
    df['y'] = (trend * weekly_seasonality * promo_spike) + noise
    df['y'] = np.maximum(df['y'], 0) # Can't have negative sales
    
    return df

def train_and_evaluate_prophet(df):
    """Trains the model using historical data AND marketing schedules."""
    train_data = df[:-30] # Hide the last 30 days to test the AI
    test_data = df[-30:]
    
    print(f"Training Prophet Model with External Regressors...")
    
    # PRD Requirement F-03: Use external regressors
    model = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=True)
    
    # We explicitly tell the AI to factor in our marketing promotions!
    model.add_regressor('is_promotion') 
    
    model.fit(train_data)
    
    # Create the future dataframe to predict the next 30 days
    future = model.make_future_dataframe(periods=30)
    
    # To predict the future, the AI needs to know if we plan to run a promotion in the future
    future['is_promotion'] = df['is_promotion'].values 
    
    print("Predicting the next 30 days of demand...")
    forecast = model.predict(future)
    
    predictions = forecast[-30:]['yhat'].values
    actuals = test_data['y'].values
    
    # Calculate the PRD Metric: Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(actuals, predictions)
    
    print("\n--- FORECAST EVALUATION ---")
    print(f"Model MAPE: {mape:.3f} (or {mape * 100:.1f}%)")
    
    if mape <= 0.10:
        print("✅ MODEL PASSED: Meets enterprise PRD threshold (MAPE <= 10%)")
    else:
        print(f"⚠️ MODEL WARNING: MAPE is {(mape * 100):.1f}%.")
        
    return model, forecast

def main():
    ts_data = generate_predictable_demand()
    model, forecast = train_and_evaluate_prophet(ts_data)
    
    # Save the predictions so our Dashboard can read them
    output_dir = 'data_pipelines/features'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = f"{output_dir}/demand_forecast.parquet"
    forecast.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"\n✅ Success! Saved 30-day forecast to {output_path}")

if __name__ == "__main__":
    main()