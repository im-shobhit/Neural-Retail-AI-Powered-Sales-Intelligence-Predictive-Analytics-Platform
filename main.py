from fastapi import FastAPI, HTTPException
import pandas as pd
import json

# Initialize our application
app = FastAPI(title="NeuralRetail API", version="1.0")

# 1. Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "NeuralRetail API"}

# 2. NEW: Forecast data endpoint
@app.get("/forecast")
def get_forecast():
    try:
        # Read the forecast CSV file
        df = pd.read_csv("data/processed/sales_forecast.csv")
        
        # Convert the Pandas DataFrame into a JSON format that APIs can send
        # 'records' orientation turns rows into a list of dictionaries
        data = df.to_dict(orient="records")
        return data
        
    except FileNotFoundError:
        # If the file doesn't exist, throw a proper HTTP error
        raise HTTPException(status_code=404, detail="Forecast data not found. Run train_forecast.py first.")