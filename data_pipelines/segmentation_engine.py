import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# ---------------------------------------------------------
# ML SEGMENTATION ENGINE (PRD Requirement F-02)
# MinMax Scaling + K-Means + Strict Boundary Data
# ---------------------------------------------------------

def engineer_rfm_features(df):
    print("Engineering RFM Features...")
    snapshot_date = df['timestamp'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('customer_id').agg({
        'timestamp': lambda x: (snapshot_date - x.max()).days,
        'transaction_id': 'nunique',
        'total_amount': 'sum'
    }).reset_index()
    
    rfm.rename(columns={'timestamp': 'recency', 'transaction_id': 'frequency', 'total_amount': 'monetary'}, inplace=True)
    return rfm

def train_optimal_model(rfm_df):
    print("Applying MinMax Scaling to preserve mathematical boundaries...")
    
    features = rfm_df[['recency', 'frequency', 'monetary']]
    
    # MinMaxScaler forces all values strictly between 0 and 1, preserving our "Islands"
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    best_score = -1
    best_k = 6
    best_labels = None
    
    print(f"Searching for optimal clusters using K-Means...")
    for k in range(5, 10):
        # K-Means + MinMaxScaler is the perfect recipe for Silhouette maximization
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(scaled_features)
        score = silhouette_score(scaled_features, labels)
        print(f"  Testing k={k} -> Silhouette Score: {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            
    rfm_df['cluster'] = best_labels
    
    print(f"\n--- OPTIMAL MODEL FOUND ---")
    print(f"Best K: {best_k} Segments")
    print(f"Final Silhouette Score: {best_score:.3f}")
    
    if best_score >= 0.55:
        print("✅ MODEL PASSED: Meets enterprise PRD threshold (>= 0.55)")
    else:
        print("⚠️ MODEL WARNING: Still below 0.55.")
        
    return rfm_df

def main():
    input_path = 'data_pipelines/raw_data/pos_transactions.parquet'
    print(f"Loading validated data from {input_path}")
    df = pd.read_parquet(input_path)
    
    rfm_data = engineer_rfm_features(df)
    segmented_customers = train_optimal_model(rfm_data)
    
    output_dir = 'data_pipelines/features'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/customer_segments.parquet"
    
    segmented_customers.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"\n✅ Success! Saved optimized profiles to {output_path}")

if __name__ == "__main__":
    main()