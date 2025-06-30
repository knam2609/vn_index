import pandas as pd

# Load the data
df = pd.read_csv('raw_data/vn_index_data/hose_historical_data.csv')

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
    df.to_csv('ready_data/vn_index_data/cleaned_vn_index_data.csv', index=False)
    print(df)

    print("✅ Data preprocessing completed and saved to 'cleaned_vn_index_data.csv'")

    return df

if __name__ == "__main__":
    # Load the data
    df = pd.read_csv('raw_data/vn_index_data/hose_historical_data.csv')
    process_data(df)