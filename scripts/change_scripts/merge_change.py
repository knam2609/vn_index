import pandas as pd

# Load the datasets
change_vn_index_df = pd.read_csv("raw_data/change_data/change_vn_index.csv")
merged_vn_index_df = pd.read_csv("ready_data/vn_index_data/vn_index_merged_data.csv")
change_vn_30_df = pd.read_csv("raw_data/change_data/change_vn_30.csv")
merged_vn_30_df = pd.read_csv("ready_data/vn_30_data/vn_30_merged_data.csv")
change_vn_30f1_df = pd.read_csv("raw_data/change_data/change_vn_30f1.csv")
merged_vn_30f1_df = pd.read_csv("ready_data/vn_30f1_data/vn_30f1_merged_data.csv")

# Perform an inner merge:
change_vn_index = pd.merge(change_vn_index_df, merged_vn_index_df, how='inner', on='Date')
change_vn_30 = pd.merge(change_vn_30_df, merged_vn_30_df, how='inner', on='Date')
change_vn_30f1 = pd.merge(change_vn_30f1_df, merged_vn_30f1_df, how='inner', on='Date')

change_vn_index.to_csv("ready_data/change_data/change_vn_index.csv", index=False)
change_vn_30.to_csv("ready_data/change_data/change_vn_30.csv", index=False)
change_vn_30f1.to_csv("ready_data/change_data/change_vn_30f1.csv", index=False)


