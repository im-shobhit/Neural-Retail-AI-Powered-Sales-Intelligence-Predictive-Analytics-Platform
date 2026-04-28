import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def build_rfm_clusters():
    print("Loading cleaned data...")
    df = pd.read_csv("data/processed/cleaned_retail.csv")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    price_col = 'Price' if 'Price' in df.columns else 'UnitPrice'
    customer_col = 'Customer ID' if 'Customer ID' in df.columns else 'CustomerID'
    
    df['TotalPrice'] = df['Quantity'] * df[price_col]
    
    print("Calculating base RFM metrics...")
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby(customer_col).agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days, 
        'Invoice': 'nunique',                                    
        'TotalPrice': 'sum'                                      
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    rfm = rfm[rfm['Monetary'] > 0]
    
    print("Training K-Means AI Model...")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # --- THE BACKEND FIX: DYNAMIC SORTING ---
    print("Dynamically mapping AI clusters to business rules...")
    
    # Calculate the average Recency, Frequency, and Monetary value for each AI cluster
    cluster_means = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    
    # Sort the clusters purely by how much money they spend on average (Highest to Lowest)
    cluster_means = cluster_means.sort_values(by='Monetary', ascending=False)
    
    # Extract the sorted ID numbers
    ordered_ids = cluster_means.index.tolist()
    
    # Safely assign names based on their actual mathematical rank
    dynamic_names = {
        ordered_ids[0]: 'VIPs',             # Highest spenders
        ordered_ids[1]: 'Loyal/Active',     # Second highest
        ordered_ids[2]: 'New/Occasional',   # Third highest
        ordered_ids[3]: 'At-Risk'           # Lowest spenders
    }
    
    rfm['Segment'] = rfm['Cluster'].map(dynamic_names)
    # ----------------------------------------
    
    output_path = "data/processed/rfm_clusters.csv"
    rfm.to_csv(output_path, index=False)
    
    print(f"\n✅ AI Customer Segments successfully saved to: {output_path}")
    print("\nSneak peek at your FIXED AI-assigned segments:")
    print(rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment']].head())

if __name__ == "__main__":
    build_rfm_clusters()