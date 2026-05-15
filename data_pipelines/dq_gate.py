import pandas as pd
import great_expectations as gx
import sys

# ---------------------------------------------------------
# DATA QUALITY GATE (PRD Requirement F-01)
# Ensures zero silent failures before feature engineering.
# ---------------------------------------------------------

def run_quality_gate(file_path):
    print(f"Loading data from {file_path}...")
    
    # Load the Parquet file into Pandas
    df = pd.read_parquet(file_path)
    
    # Wrap the dataframe in a Great Expectations dataset
    gx_df = gx.from_pandas(df)
    
    print("\nRunning Data Quality Expectations...")
    
    # ---------------------------------------------------------
    # EXPECTATION 1: No negative prices (ERP Glitch Prevention)
    # ---------------------------------------------------------
    res_price = gx_df.expect_column_values_to_be_between(
        column="unit_price", min_value=0.01, max_value=1000.00
    )
    
    # ---------------------------------------------------------
    # EXPECTATION 2: Store IDs must match our exact known list
    # ---------------------------------------------------------
    res_store = gx_df.expect_column_values_to_be_in_set(
        column="store_id", value_set=['STORE_A', 'STORE_B', 'WEB_01', 'WEB_02']
    )
    
    # ---------------------------------------------------------
    # EXPECTATION 3: No missing (NULL) transaction IDs
    # ---------------------------------------------------------
    res_txn = gx_df.expect_column_values_to_not_be_null(column="transaction_id")

    # ---------------------------------------------------------
    # EVALUATE RESULTS
    # ---------------------------------------------------------
    results = [res_price.success, res_store.success, res_txn.success]
    dq_score = (sum(results) / len(results)) * 100
    
    print("-" * 40)
    print(f"Price Validation: {'✅ PASS' if res_price.success else '❌ FAIL'}")
    print(f"Store Validation: {'✅ PASS' if res_store.success else '❌ FAIL'}")
    print(f"Null Validation:  {'✅ PASS' if res_txn.success else '❌ FAIL'}")
    print("-" * 40)
    
    print(f"Final Data Quality Score: {dq_score:.1f}%")
    
    # PRD demands DQ Score >= 98%
    if dq_score >= 98.0:
        print("✅ QUALITY GATE PASSED. Data is safe for ML Pipeline.")
    else:
        print("🚨 QUALITY GATE FAILED. Halting pipeline to prevent silent failures.")
        sys.exit(1) # This tells Airflow/CI-CD that the pipeline crashed!

if __name__ == "__main__":
    target_file = 'data_pipelines/raw_data/pos_transactions.parquet'
    run_quality_gate(target_file)