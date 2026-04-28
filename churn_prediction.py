import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_advanced_churn_model():
    print("Loading raw transaction history for Feature Engineering...")
    df = pd.read_csv("data/processed/cleaned_retail.csv")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    price_col = 'Price' if 'Price' in df.columns else 'UnitPrice'
    customer_col = 'Customer ID' if 'Customer ID' in df.columns else 'CustomerID'
    df['TotalPrice'] = df['Quantity'] * df[price_col]
    
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    print("Engineering advanced behavioral features...")
    # Extract deep transaction histories per customer
    customer_data = df.groupby(customer_col).agg({
        'InvoiceDate': ['max', 'min'],
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    
    # Flatten the column names
    customer_data.columns = ['CustomerID', 'LastPurchaseDate', 'FirstPurchaseDate', 'Frequency', 'Monetary']
    
    # FEATURE 1: Recency (Target Variable Base)
    customer_data['Recency'] = (snapshot_date - customer_data['LastPurchaseDate']).dt.days
    
    # FEATURE 2: Tenure (Days as a customer)
    customer_data['Tenure'] = (customer_data['LastPurchaseDate'] - customer_data['FirstPurchaseDate']).dt.days
    
    # FEATURE 3: Purchase Pace (Average days between orders)
    # Replaced 0 with 1 to prevent division by zero errors for one-time buyers
    customer_data['PurchasePace'] = customer_data['Tenure'] / customer_data['Frequency'].replace(0, 1)
    
    # FEATURE 4: Average Order Value
    customer_data['AOV'] = customer_data['Monetary'] / customer_data['Frequency']
    
    # THE TARGET: Has it been more than 90 days?
    customer_data['Churn'] = (customer_data['Recency'] > 90).astype(int)
    
    # Clean anomalies
    customer_data = customer_data[customer_data['Monetary'] > 0]
    
    print("Training Upgraded Random Forest AI...")
    # Note: We feed it our new features, but keep Recency hidden so it can't cheat!
    X = customer_data[['Frequency', 'Monetary', 'AOV', 'Tenure', 'PurchasePace']]
    y = customer_data['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Upgraded AI parameters: 200 trees, balanced class weights for better pattern recognition
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    print("Testing AI accuracy...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print("\n" + "="*45)
    print(f"✅ Advanced AI Churn Model Trained Successfully!")
    print(f"🎯 New Model Accuracy: {accuracy * 100:.2f}%")
    print("="*45)
    
    print("\nDetailed Performance Report:")
    print(classification_report(y_test, predictions, target_names=['Retained (0)', 'Churned (1)']))

if __name__ == "__main__":
    train_advanced_churn_model()