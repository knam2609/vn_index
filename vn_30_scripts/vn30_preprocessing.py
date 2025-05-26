import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("vn_30_data/vn_30_historical_data.csv")  # Update path if needed

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Convert numeric columns to float
numeric_cols = ['Price', 'Open', 'High', 'Low']
for col in numeric_cols:
    df[col] = df[col].str.replace(',', '').astype(float)

# Rename columns
df.rename(columns={
    'Price': 'VN_30_Close',
    'Vol.': 'Volume'
}, inplace=True)

# Parse volume strings like '557.83M' or '107.03K'
def parse_volume(vol_str):
    if isinstance(vol_str, str):
        vol_str = vol_str.replace(',', '')
        if 'M' in vol_str:
            return float(vol_str.replace('M', '')) * 1_000_000
        elif 'K' in vol_str:
            return float(vol_str.replace('K', '')) * 1_000
    return np.nan

df['Volume'] = df['Volume'].apply(parse_volume)

# Drop 'Change %' column
if 'Change %' in df.columns:
    df.drop(columns=['Change %'], inplace=True)

# Drop rows with any missing (NaN) values
df.dropna(inplace=True)

# Drop duplicate rows (if any)
df.drop_duplicates(inplace=True)

# Sort by date ascending
df.sort_values(by='Date', inplace=True)

# Extract percentage from Change column and convert to decimal
df['Change'] = df['VN_30_Close'].pct_change()
df = df.fillna(0)

# keep only rows where |Change| ≤ 0.03
df = df[df['Change'].abs() <= 0.03]
df.reset_index(drop=True, inplace=True)

# (Optional) Save cleaned data
df.to_csv("ready_data/cleaned_vn_30_data.csv", index=False)

print("✅ VN30 historical data cleaned and saved to 'cleaned_vn_30_data.csv'")

print(df)


