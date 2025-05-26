# Re-import needed modules after reset
import pandas as pd

# Reload the uploaded file
df = pd.read_csv("vn_index_data/hose_historical_data.csv")

# Extract Date and Change columns
df_change = df[['Date', 'Change']].copy()

# Convert 'Date' to datetime type
df_change['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

df_change = df_change.sort_values('Date')

# Save the cleaned DataFrame to a new CSV
df_change.to_csv('ready_data/change.csv', index=False)


