import sys
import os

# Add the `scripts/` directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from external_data_scripts.external_preprocessing import preprocess_file
import pandas as pd

df = preprocess_file('raw_data/asx_data/ASX_200_historical_data.csv')
print(df)

if df is not None:
    df.to_csv('ready_data/asx_data/cleaned_asx_200_data.csv', index=False) 