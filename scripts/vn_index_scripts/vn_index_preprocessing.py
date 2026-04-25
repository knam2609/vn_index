import pandas as pd
from scripts.s3_scripts.read_write_to_s3 import read_csv_from_s3, write_df_to_s3

BUCKET = "vn-index"
RAW_KEY = "raw_data/vn_index_data/hose_historical_data.csv"
CLEANED_KEY = "ready_data/vn_index_data/cleaned_vn_index_data.csv"
REQUIRED_COLUMNS = ["Date", "VN-INDEX", "Total Volume", "Total Value"]
OUTPUT_COLUMNS = ["Date", "VN_Index_Close", "Total Volume", "Total Value"]

def remove_outliers(df):
    # Extract percentage from Change column and convert to decimal
    df['Change'] = df[df.columns[1]].pct_change()  # Calculate percentage change
    df = df.fillna(0)  # Fill NaN values with 0

    # keep only rows where |Change| ≤ 0.03
    df = df[df['Change'].abs() <= 0.03]
    df.reset_index(drop=True, inplace=True)  # Reset index after filtering
    return df

def convert_to_numeric(value):
    if pd.isna(value):
        return pd.NA

    if isinstance(value, str):
        value = value.strip()
        if not value or value == "---":
            return pd.NA

        value = value.replace(',', '')
        if value.endswith(' bil'):
            return float(value.replace(' bil', '')) * 1e9
        if value.endswith(' mil'):
            return float(value.replace(' mil', '')) * 1e6

    return float(value)

def process_data(df):
    df = df.copy()

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required VN-Index columns: {', '.join(missing_columns)}")

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Keep only forecasting inputs before dropping nulls. Stockbiz can leave
    # ancillary columns blank, and those should not discard otherwise valid rows.
    df = df[REQUIRED_COLUMNS].copy()
    df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)
    df.replace("---", pd.NA, inplace=True)
    df.dropna(subset=REQUIRED_COLUMNS, inplace=True)

    # Convert 'Date' to datetime type
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors="coerce")

    # Apply the conversion to all relevant columns
    for col in ['VN-INDEX', 'Total Volume', 'Total Value']:
        df[col] = df[col].apply(convert_to_numeric)

    df.rename(columns={'VN-INDEX': 'VN_Index_Close'}, inplace=True)
    df.dropna(subset=OUTPUT_COLUMNS, inplace=True)

    if df.empty:
        raise ValueError(
            "No valid VN-Index rows remain after preprocessing; "
            "refusing to overwrite cleaned_vn_index_data.csv"
        )

    df = df.sort_values('Date')

    # Save the cleaned data to a new CSV file
    write_df_to_s3(df, BUCKET, CLEANED_KEY)
    print(df)

    print("✅ Data preprocessing completed and saved to 'cleaned_vn_index_data.csv'")

    return df

if __name__ == "__main__":
    # Load the data
    df = read_csv_from_s3(BUCKET, RAW_KEY)
    # df = remove_outliers(df)
    process_data(df)
