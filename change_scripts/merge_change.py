import pandas as pd

# Load the datasets
change_df = pd.read_csv("ready_data/change.csv")
vn_index_df = pd.read_csv("ready_data/cleaned_vn_index_data.csv")
external_df = pd.read_csv("ready_data/vn_index_external_data.csv")
merged_df = pd.read_csv("ready_data/vn_index_merged_data.csv")

# Perform an inner merge:
change_vn_index = pd.merge(change_df, vn_index_df, how='inner', on='Date')
change_external = pd.merge(change_df, external_df, how='inner', on='Date')
change_merged = pd.merge(change_df, merged_df, how='inner', on='Date')

change_vn_index.to_csv("ready_data/change_vn_index.csv", index=False)
change_external.to_csv("ready_data/change_external.csv", index=False)
change_merged.to_csv("ready_data/change_merged.csv", index=False)


