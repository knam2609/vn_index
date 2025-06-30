import pandas as pd

# Load the two datasets
vn30_df = pd.read_csv("ready_data/vn_30_data/cleaned_vn_30_data.csv")
external_df = pd.read_csv("ready_data/external_data/cleaned_external_data")
list_df = pd.read_csv("ready_data/vn_30_list_data/cleaned_vn30_list_data.csv")


# Perform an inner merge: VN30 data first, merged with list and external info
merged_df = pd.merge(vn30_df, list_df, how='outer', on='Date')
merged_df = pd.merge(merged_df, external_df, how='outer', on='Date')


# Get list of other columns (all columns not part of vn30_df)
other_columns = list(set(merged_df.columns) - set(vn30_df.columns))

def input_missing(df, other_columns, target_column):
    # Fill missing values in other columns using proportional target changes
    for col in other_columns:
        for i in reversed(range(len(df) - 1)):
            if pd.isna(df.loc[i, col]) and not pd.isna(df.loc[i + 1, col]):
                try:
                    scale = df.loc[i, target_column] / df.loc[i + 1, target_column] # type: ignore
                    df.loc[i, col] = df.loc[i + 1, col] * scale # type: ignore
                except:
                    pass  # skip if division is invalid

print(merged_df)

merged_df.to_csv('ready_data/vn_30_merged_data.csv', index=False)


