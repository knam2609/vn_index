import pandas as pd

# Load CSV
df_vn30f1 = pd.read_csv("raw_data/vn_30_data/vn_30f1_historical_data.csv")[['Date', 'Close']]
df_vn30f1['Date'] = pd.to_datetime(df_vn30f1['Date'], format='%d-%m-%y')
df_vn30f1.sort_values('Date', inplace=True)
df_vn30f1.rename(columns={'Close': 'VN_30F1_Close'}, inplace=True)
# Extract percentage from Change column and convert to decimal
df_vn30f1['Change'] = df_vn30f1['VN_30F1_Close'].pct_change()
df_vn30f1 = df_vn30f1.fillna(0)

# keep only rows where |Change| ≤ 0.03
df_vn30f1 = df_vn30f1[df_vn30f1['Change'].abs() <= 0.03][['Date','VN_30F1_Close']]
df_vn30f1.reset_index(drop=True, inplace=True)

df_vn30 = pd.read_csv('ready_data/vn_30_data/cleaned_vn_30_data.csv')[['Date', 'VN_30_Close']]
df_vn30.rename(columns={'VN_30_Close': 'VN_30F1_Close'}, inplace=True)

# Concatenate historical data of VN30 to VN30F1
df = pd.concat([df_vn30f1, df_vn30], ignore_index=True).iloc[:-len(df_vn30f1)]
df['Date'] = pd.to_datetime(df['Date'])

df.sort_values('Date', inplace=True)
df.reset_index(inplace=True, drop=True)
df.to_csv('ready_data/vn_30_data/cleaned_vn_30f1_data.csv', index=False)
print(df)

print("✅ VN30F1 historical data cleaned and saved to 'cleaned_vn_30f1_data.csv'")



