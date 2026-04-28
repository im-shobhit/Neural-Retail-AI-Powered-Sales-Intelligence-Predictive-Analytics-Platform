import pandas as pd

def clean_retail_data():
    print("Loading raw data...")
    df = pd.read_csv("data/raw/online_retail_II.csv")
    
    # 1. Drop rows where Customer ID is missing
    print("Dropping missing Customer IDs...")
    df = df.dropna(subset=['Customer ID'])
    
    # 2. Filter out cancelled orders (Invoices starting with 'C')
    print("Removing cancelled orders...")
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    
    # 3. Convert InvoiceDate to a proper datetime object
    print("Formatting dates...")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # 4. Save the cleaned data
    output_path = "data/processed/cleaned_retail.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Clean data saved to: {output_path}")
    print(f"Remaining rows after cleaning: {len(df)}")

if __name__ == "__main__":
    clean_retail_data()