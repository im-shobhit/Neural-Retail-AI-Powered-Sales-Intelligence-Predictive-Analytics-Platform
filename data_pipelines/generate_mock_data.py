import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_transactional_data():
    print("Generating transaction records with PERFECT Persona separation...")
    np.random.seed(42)
    
    customer_ids = np.arange(1000, 5000) # 4000 distinct customers
    # Distribute customers across 6 personas
    personas = np.random.choice([0, 1, 2, 3, 4, 5], size=len(customer_ids))
    
    dates = []
    c_ids = []
    quantities = []
    prices = []
    tx_ids = []
    tx_counter = 1
    
    end_date = datetime.now()
    
    # THE FIX: Loop through each customer and spawn the correct number of transactions
    for cid, p in zip(customer_ids, personas):
        
        # 0: Whales (Recency 1-10, Freq 20-30, High Spend)
        if p == 0:   
            freq = np.random.randint(20, 31)
            for _ in range(freq):
                dates.append(end_date - timedelta(days=np.random.randint(1, 10)))
                quantities.append(np.random.randint(5, 10))
                prices.append(np.random.uniform(100, 200))
                c_ids.append(cid)
                tx_ids.append(f"TXN-{tx_counter:07d}")
                tx_counter += 1
                
        # 1: Churned (Recency 300-365, Freq 1-2, Low Spend)
        elif p == 1: 
            freq = np.random.randint(1, 3)
            for _ in range(freq):
                dates.append(end_date - timedelta(days=np.random.randint(300, 365)))
                quantities.append(1)
                prices.append(np.random.uniform(10, 20))
                c_ids.append(cid)
                tx_ids.append(f"TXN-{tx_counter:07d}")
                tx_counter += 1
                
        # 2: Regulars (Recency 30-60, Freq 8-12, Mid Spend)
        elif p == 2: 
            freq = np.random.randint(8, 13)
            for _ in range(freq):
                dates.append(end_date - timedelta(days=np.random.randint(30, 60)))
                quantities.append(np.random.randint(2, 5))
                prices.append(np.random.uniform(40, 80))
                c_ids.append(cid)
                tx_ids.append(f"TXN-{tx_counter:07d}")
                tx_counter += 1
                
        # 3: Deal Hunters (Recency 10-20, Freq 15-20, Tiny Spend)
        elif p == 3: 
            freq = np.random.randint(15, 21)
            for _ in range(freq):
                dates.append(end_date - timedelta(days=np.random.randint(10, 20)))
                quantities.append(np.random.randint(3, 6))
                prices.append(np.random.uniform(5, 15))
                c_ids.append(cid)
                tx_ids.append(f"TXN-{tx_counter:07d}")
                tx_counter += 1
                
        # 4: Seasonal (Recency 150-180, Freq 5-8, High Spend)
        elif p == 4: 
            freq = np.random.randint(5, 9)
            for _ in range(freq):
                dates.append(end_date - timedelta(days=np.random.randint(150, 180)))
                quantities.append(np.random.randint(3, 7))
                prices.append(np.random.uniform(100, 150))
                c_ids.append(cid)
                tx_ids.append(f"TXN-{tx_counter:07d}")
                tx_counter += 1
                
        # 5: Window Shoppers (Recency 60-90, Freq 1-2, Tiny Spend)
        elif p == 5: 
            freq = np.random.randint(1, 3)
            for _ in range(freq):
                dates.append(end_date - timedelta(days=np.random.randint(60, 90)))
                quantities.append(1)
                prices.append(np.random.uniform(5, 10))
                c_ids.append(cid)
                tx_ids.append(f"TXN-{tx_counter:07d}")
                tx_counter += 1

    num_records = len(tx_ids)
    data = {
        'transaction_id': tx_ids,
        'timestamp': dates,
        'customer_id': c_ids,
        'sku_id': np.random.randint(100, 300, num_records),
        'store_id': np.random.choice(['STORE_A', 'STORE_B', 'WEB_01', 'WEB_02'], num_records),
        'quantity': quantities,
        'unit_price': np.round(prices, 2),
        'is_promotion': np.random.choice([0, 1], num_records, p=[0.8, 0.2]),
        'weather_condition': np.random.choice(['Sunny', 'Rain', 'Cloudy', 'Snow'], num_records)
    }
    
    df = pd.DataFrame(data)
    df['total_amount'] = df['quantity'] * df['unit_price']
    df.loc[df['is_promotion'] == 1, 'total_amount'] *= 0.85 
    
    return df

def main():
    os.makedirs('data_pipelines/raw_data', exist_ok=True)
    df_transactions = generate_transactional_data() 
    output_path = 'data_pipelines/raw_data/pos_transactions.parquet'
    df_transactions.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"✅ Success! Saved {len(df_transactions)} mathematically distinct rows to {output_path}")

if __name__ == "__main__":
    main()