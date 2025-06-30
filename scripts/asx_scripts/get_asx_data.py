import sys
import os
from datetime import datetime

# Add the `scripts/` directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from external_data_scripts.get_external_data import download_data
import pandas as pd

# 📅 Define start and end dates
start_date = "1984-01-03"
end_date = datetime.today().strftime("%Y-%m-%d")
print(f"Downloading data up to {end_date}")

# Download Microsoft stock data
output_dir = 'raw_data/asx_data/'
download_data('ASX', '^AORD', output_dir, start_date, end_date)
df = pd.read_csv('raw_data/asx_data/ASX_historical_data.csv')
print(df)