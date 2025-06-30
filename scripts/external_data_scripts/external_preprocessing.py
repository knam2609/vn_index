import os
import pandas as pd

# ✅ Function to preprocess a single CSV file
def preprocess_file(file_path):
    try:
        # Read CSV, skipping first two rows to fix column names issue
        df = pd.read_csv(file_path)

        # Rename columns
        df.rename(columns={'close': 'Close'}, inplace=True)
        df.rename(columns={'date': 'Date'}, inplace=True)

        # Ensure 'Date' is parsed correctly
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', utc=True)
        df["Date"] = df["Date"].dt.tz_localize(None).dt.date

        df = df.drop_duplicates('Date')
        
        df = df[['Date', 'Close']]

        # Rename columns: Add file name as prefix to avoid conflicts
        file_name = os.path.basename(file_path).replace("_historical_data.csv", "")
        df.rename(columns={'Close': f"{file_name}_Close"}, inplace=True)
        
        if df.shape[0] < 4500:
            print(f"INSUFFICIENT DATA FOR {file_name}")
            return None

        print(f"✅ Processed: {file_name}")
        return df
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

def main():
    # 📂 Define the folder containing all CSV files
    data_folder = "raw_data/external_data"

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
            merged_df = pd.merge(merged_df, df, on="Date", how="outer")

        # ✅ Handle missing values (Forward Fill & Backward Fill)
        merged_df.fillna(method='ffill', inplace=True)  # type: ignore # Forward fill
        merged_df.fillna(method='bfill', inplace=True)  # type: ignore # Backward fill (if needed)

        merged_df = merged_df.sort_values(by='Date')

        # ✅ Save preprocessed data
        merged_df.to_csv("ready_data/external_data/cleaned_external_data.csv", index=False)
        print("🎉 Preprocessing complete! Data saved to `cleaned_external_data.csv`")
        print(merged_df)
    else:
        print("⚠️ No valid CSV files found in the folder.")

if __name__ == "__main__":
    main()