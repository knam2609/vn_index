import os
import pandas as pd

# 📂 Define the folder containing all CSV files
data_folder = "vn_30_list"

# Function to convert values to numeric, handling 'mil', 'bil', and commas
def convert_to_numeric(value):
    if isinstance(value, str):
        value = value.replace(',', '')  # Remove commas
        if ' bil' in value:  
            return float(value.replace(' bil', '')) * 1e9  # Convert 'bil' to numeric
        elif ' mil' in value:
            return float(value.replace(' mil', '')) * 1e6  # Convert 'mil' to numeric
    return float(value)  # Convert remaining values

# ✅ Function to preprocess a single CSV file
def preprocess_file(file_path):
    try:
        # Read CSV, skipping first two rows to fix column names issue
        df = pd.read_csv(file_path)

        # Ensure 'Date' is parsed correctly
        df['Date'] = pd.to_datetime(df['Date'])

        df.columns = (
            df.columns
            .str.replace(r'\s*\(.*\)', '', regex=True)  # remove anything in parentheses
            .str.replace(' ', '_')                       # replace spaces with underscores
        )
        
        file_name = os.path.basename(file_path).replace("_historical_data.csv", "")

        for col in df.columns:
            if col == 'Date':
                continue
            elif col == 'Index':
                df.drop(columns=['Index'], inplace=True)
            else:
                if col == 'Close':
                    df[col] = df[col].astype(float)
                else:
                    df[col].apply(convert_to_numeric)
                    
                # Rename columns: Add file name as prefix to avoid conflicts
                df.rename(columns={col: f"{file_name}_{col}"}, inplace=True)

        print(f"✅ Processed: {file_name}")
        return df
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

# ✅ Read all CSV files in the folder
all_dataframes = []
for file in os.listdir(data_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(data_folder, file)
        df = preprocess_file(file_path)
        if df is not None:
            all_dataframes.append(df)

# ✅ Merge all data on Date
if all_dataframes:
    merged_df = all_dataframes[0]
    for df in all_dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on="Date", how="inner")

    # ✅ Handle missing values (Forward Fill & Backward Fill)
    merged_df.fillna(method='ffill', inplace=True)  # type: ignore # Forward fill
    merged_df.fillna(method='bfill', inplace=True)  # type: ignore # Backward fill (if needed)

    merged_df = merged_df.sort_values(by='Date')

    # ✅ Save preprocessed data
    merged_df.to_csv("ready_data/cleaned_vn30_list_data.csv", index=False)
    print("🎉 Preprocessing complete! Data saved to `cleaned_vn30_list_data.csv`")
else:
    print("⚠️ No valid CSV files found in the folder.")


