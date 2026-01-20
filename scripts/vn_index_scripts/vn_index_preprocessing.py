import pandas as pd
from scripts.s3_scripts.read_write_to_s3 import read_csv_from_s3, write_df_to_s3

# Load the data
df = read_csv_from_s3("vn-index", "raw_data/vn_index_data/hose_historical_data.csv")

def remove_outliers(df):
    # Extract percentage from Change column and convert to decimal
    df['Change'] = df[df.columns[1]].pct_change()  # Calculate percentage change
    df = df.fillna(0)  # Fill NaN values with 0

    # keep only rows where |Change| ≤ 0.03
    df = df[df['Change'].abs() <= 0.03]
    df.reset_index(drop=True, inplace=True)  # Reset index after filtering
    return df

def process_data(df):
    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Remove rows with missing values
    # df.replace('---', pd.NA, inplace=True)  
    df.dropna(inplace=True)

    df = df[['Date', 'VN-INDEX', 'Total Volume', 'Total Value']]

    # Convert 'Date' to datetime type
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

    print(df.columns)

    # Function to convert values to numeric, handling 'mil', 'bil', and commas
    def convert_to_numeric(value):
        if isinstance(value, str):
            value = value.replace(',', '')  # Remove commas
            if ' bil' in value:  
                return float(value.replace(' bil', '')) * 1e9  # Convert 'bil' to numeric
            elif ' mil' in value:
                return float(value.replace(' mil', '')) * 1e6  # Convert 'mil' to numeric
        return float(value)  # Convert remaining values

    # Apply the conversion to all relevant columns
    columns_to_convert = ['Total Volume', 'Total Value']

    for col in columns_to_convert:
        df[col] = df[col].apply(convert_to_numeric)

    # Convert other columns to numerical types
    df['VN-INDEX'] = df['VN-INDEX'].astype(float)
    df.rename(columns={'VN-INDEX': 'VN_Index_Close'}, inplace=True)

    df = df.sort_values('Date')

    # Save the cleaned data to a new CSV file
    write_df_to_s3(df, "vn-index", "ready_data/vn_index_data/cleaned_vn_index_data.csv")
    print(df)

    print("✅ Data preprocessing completed and saved to 'cleaned_vn_index_data.csv'")

    return df

if __name__ == "__main__":
    # Load the data
    df = read_csv_from_s3("vn-index", "raw_data/vn_index_data/hose_historical_data.csv")
    # df = remove_outliers(df)
    process_data(df)