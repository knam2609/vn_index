import time
import random
from datetime import datetime
import os

from yahooquery import Ticker

# 📅 Define start and end dates
start_date = "2000-07-28"
end_date = datetime.today().strftime("%Y-%m-%d")
print(f"Downloading data up to {end_date}")

# 📊 Financial & Commodity Data (from Yahoo via yahooquery)
assets = {
    "S&P_500": "^GSPC",
    "DJIA": "^DJI",
    "NASDAQ": "^IXIC",
    "Shanghai": "000001.SS",
    "Hang_Seng_Index": "^HSI",
    "KOSPI": "^KS11",
    "Taiwan_Weighted_Index": "^TWII",
    "FTSE_100": "^FTSE",
    "Brent_Crude_Oil": "BZ=F",
    "WTI_Crude_Oil": "CL=F",
    "Gold": "GC=F",
    "LNG": "NG=F",
    "Copper": "HG=F",
    "Aluminum": "ALI=F",
    "Iron_Ore": "TIOc1",
    "USDVND": "USDVND=X",
    "CNYVND": "CNYVND=X",
    "JPYVND": "JPYVND=X",
    "KRWVND": "KRWVND=X",
    "EURVND": "EURVND=X",
    "US_10Y_Treasury_Yield": "^TNX"
}

# Ensure output directory exists
output_dir = "external_data"
os.makedirs(output_dir, exist_ok=True)

def download_data(asset_name, ticker_symbol, max_retries=3):
    """Download historical data using yahooquery and save to CSV."""
    retries = 0
    while retries < max_retries:
        try:
            print(f"📥 Downloading data for {asset_name} ({ticker_symbol})...")
            ticker = Ticker(ticker_symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")
            
            # yahooquery returns a DataFrame with symbol as level-0 if multiple; select symbol
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs(ticker_symbol, level=0, axis=1)
            
            if df.empty:
                print(f"⚠️ No data available for {asset_name}, skipping...")
                return None
            
            # Save to CSV with sanitized filename
            safe_name = asset_name.replace("/", "_").replace("=", "_")
            file_name = os.path.join(output_dir, f"{safe_name}_historical_data.csv")
            df.to_csv(file_name)
            print(f"✅ Data saved to {file_name}")
            
            # 🕒 Random delay to avoid rate limiting
            time.sleep(random.uniform(2, 5))
            return df

        except Exception as e:
            print(f"⚠️ Error downloading {asset_name}: {e}")
            retries += 1
            time.sleep(10)  # wait before retrying
    
    print(f"❌ Failed to download {asset_name} after {max_retries} retries.")
    return None

# Download and save data for all assets
import pandas as pd  # needed for MultiIndex check in download_data
for name, sym in assets.items():
    download_data(name, sym)

print("🎉 All available data processing complete!")

df = pd.read_csv('external_data/S&P_500_historical_data.csv')
print(df)


