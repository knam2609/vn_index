import pandas as pd
import numpy as np

def remove_all_zeros(df, tolerance=1e-8):
    # Select only numeric columns to check
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_drop = []
    for col in numeric_cols:
        if np.isclose(df[col], 0, atol=tolerance).all():
            print(f"All values in {col} are effectively zero")
            cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop)


# 📂 File paths
vn_index_file = "ready_data/cleaned_vn_index_data.csv"  
external_data_file = "ready_data/cleaned_external_data.csv"    # Processed external data

# ✅ Load VN-Index Data
vn_index_df = pd.read_csv(vn_index_file, parse_dates=["Date"])
vn_index_df.rename(columns={"VN-INDEX": "VN_Index_Close"}, inplace=True)  # Rename column

# ✅ Load External Data
external_df = pd.read_csv(external_data_file, parse_dates=["Date"])

# ✅ Merge using Left Join (VN-Index as reference)
merged_df = pd.merge(vn_index_df, external_df, on="Date", how="inner")
vn_index_external_df = pd.merge(vn_index_df[['Date', 'VN_Index_Close']], external_df, on="Date", how="inner")

merged_df = remove_all_zeros(merged_df)
vn_index_external_df = remove_all_zeros(vn_index_external_df)

# ✅ Sort values by Date
merged_df = merged_df.sort_values(by="Date")
vn_index_external_df = vn_index_external_df.sort_values(by="Date")

# ✅ Save merged dataset
merged_df.to_csv("ready_data/vn_index_merged_data.csv", index=False)
print("🎉 Merging complete! Data saved to `vn_index_merged_data.csv`")

vn_index_external_df.to_csv("ready_data/vn_index_external_data.csv", index=False)
print("🎉 Merging complete! Data saved to `vn_index_external_data.csv`")

print(merged_df)


