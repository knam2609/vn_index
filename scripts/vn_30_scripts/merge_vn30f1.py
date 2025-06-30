import pandas as pd
from .merge_vn30 import input_missing

# Load the two datasets
vn30f1 = pd.read_csv("ready_data/vn_30_data/cleaned_vn_30f1_data.csv")
external_df = pd.read_csv("ready_data/external_data/cleaned_external_data")
list_df = pd.read_csv("ready_data/vn_30_data/cleaned_vn30_list_data.csv")


# Perform an inner merge: VN30 data first, merged with list and external info
merged_df = pd.merge(vn30f1, list_df, how='outer', on='Date')
merged_df = pd.merge(merged_df, external_df, how='outer', on='Date')


# Get list of other columns (all columns not part of vn30f1)
other_columns = list(set(merged_df.columns) - set(vn30f1.columns))

input_missing(merged_df, other_columns, 'VN_30F1_Close')

print(merged_df)

merged_df.to_csv('ready_data/vn_30_data/vn_30f1_merged_data.csv', index=False)


