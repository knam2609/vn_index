import pandas as pd

# Load the two datasets
vn30_df = pd.read_csv("ready_data/cleaned_vn_30_data.csv")
list_df = pd.read_csv("ready_data/cleaned_vn30_list_data.csv")

# Perform an inner merge: VN30 data first, merged with list info
merged_df = pd.merge(vn30_df, list_df, how='outer', on='Date')


# Get list of stock columns (all columns not part of vn30_df)
stock_columns = list(set(merged_df.columns) - set(vn30_df.columns))

# Fill missing values in stock columns using proportional VN_30_Close changes
for col in stock_columns:
    for i in reversed(range(len(merged_df) - 1)):
        if pd.isna(merged_df.loc[i, col]) and not pd.isna(merged_df.loc[i + 1, col]):
            try:
                scale = merged_df.loc[i, 'VN_30_Close'] / merged_df.loc[i + 1, 'VN_30_Close'] # type: ignore
                merged_df.loc[i, col] = merged_df.loc[i + 1, col] * scale # type: ignore
            except:
                pass  # skip if division is invalid

print(merged_df)

merged_df[:-1].to_csv('ready_data/vn_30_merged_data.csv', index=False)


