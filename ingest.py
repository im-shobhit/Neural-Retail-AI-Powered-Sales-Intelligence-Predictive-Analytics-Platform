import pandas as pd

def test_ingestion():
    print("Loading raw dataset... this might take a few seconds!")
    
    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv("data/raw/online_retail_II.csv")
    
    print("\n✅ Data loaded successfully!")
    print(f"Total rows: {len(df)}")
    print("-" * 50)
    print(df.head())

if __name__ == "__main__":
    test_ingestion()