import pandas as pd
import numpy as np

# Load CSV
df_vn30 = pd.read_csv("raw_data/vn_30_data/vn_30_historical_data.csv")  # Update path if needed

# Convert 'Date' to datetime
df_vn30['Date'] = pd.to_datetime(df_vn30['Date'], format='%d/%m/%Y')

# Convert numeric columns to float
numeric_cols = ['Price', 'Open', 'High', 'Low']
for col in numeric_cols:
    df_vn30[col] = df_vn30[col].str.replace(',', '').astype(float)

# Rename columns
df_vn30.rename(columns={
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

df_vn30['Volume'] = df_vn30['Volume'].apply(parse_volume)

# Drop 'Change %' column
if 'Change %' in df_vn30.columns:
    df_vn30.drop(columns=['Change %'], inplace=True)

df_vn30.drop(columns=['Volume'], inplace=True)

# Drop rows with any missing (NaN) values
df_vn30.dropna(inplace=True)

# Drop duplicate rows (if any)
df_vn30.drop_duplicates(inplace=True)

# Sort by date ascending
df_vn30.sort_values(by='Date', inplace=True)

# Extract percentage from Change column and convert to decimal
df_vn30['Change'] = df_vn30['VN_30_Close'].pct_change()
df_vn30 = df_vn30.fillna(0)

# keep only rows where |Change| ≤ 0.03
df_vn30 = df_vn30[df_vn30['Change'].abs() <= 0.03][['Date','VN_30_Close']]
df_vn30.reset_index(drop=True, inplace=True)

df_vn_index = pd.read_csv('ready_data/vn_index_data/cleaned_vn_index_data.csv')[['Date', 'VN_Index_Close']]
df_vn_index.rename(columns={'VN_Index_Close': 'VN_30_Close'}, inplace=True)
df_vn_index['Date'] = pd.to_datetime(df_vn_index['Date'], format='%Y-%m-%d')

# Concatenate historical data of VN30 to VN30F1
df = pd.concat([df_vn30, df_vn_index], ignore_index=True).iloc[:-len(df_vn30)]
df['Date'] = pd.to_datetime(df['Date'])

df.sort_values('Date', inplace=True)
df.reset_index(inplace=True, drop=True)
df.to_csv('ready_data/vn_30_data/cleaned_vn_30_data.csv', index=False)
print(df)

print("✅ VN30 historical data cleaned and saved to 'cleaned_vn_30_data.csv'")



