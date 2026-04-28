import pandas as pd
import numpy as np

def optimize_inventory():
    print("Loading historical sales data...")
    df = pd.read_csv("data/processed/cleaned_retail.csv")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    print("Identifying Top 10 High-Volume Products...")
    top_items = df.groupby('Description')['Quantity'].sum().nlargest(10).index
    df_top = df[df['Description'].isin(top_items)]
    
    print("Calculating Safety Stock and Reorder Points (ROP)...\n")
    lead_time_days = 7
    z_score = 1.65 
    
    inventory_targets = []
    
    for item in top_items:
        item_data = df_top[df_top['Description'] == item]
        daily_sales = item_data.groupby(item_data['InvoiceDate'].dt.date)['Quantity'].sum()
        
        avg_daily_sales = daily_sales.mean()
        std_daily_sales = daily_sales.std()
        
        # --- THE FIX: Guardrail against single-day bulk orders ---
        if pd.isna(std_daily_sales):
            std_daily_sales = 0
            
        safety_stock = z_score * std_daily_sales * np.sqrt(lead_time_days)
        reorder_point = (avg_daily_sales * lead_time_days) + safety_stock
        
        inventory_targets.append({
            'Product': item[:30], 
            'Avg Daily Sales': round(avg_daily_sales, 1),
            'Safety Stock': int(round(safety_stock, 0)),
            'Reorder Point': int(round(reorder_point, 0))
        })
        
    inventory_df = pd.DataFrame(inventory_targets)
    output_path = "data/processed/inventory_targets.csv"
    inventory_df.to_csv(output_path, index=False)
    
    print(f"✅ Optimization complete! Saved to: {output_path}\n")
    print("=== WAREHOUSE RESTOCK TARGETS ===")
    print(inventory_df.to_string(index=False))

if __name__ == "__main__":
    optimize_inventory()