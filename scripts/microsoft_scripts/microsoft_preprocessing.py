import sys
import os

# Add the `scripts/` directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from external_data_scripts.external_preprocessing import preprocess_file
import pandas as pd

df = preprocess_file('raw_data/microsoft_data/MCS_historical_data.csv')
print(df)

if df is not None:
    df.to_csv('ready_data/microsoft_data/cleaned_microsoft_data.csv', index=False) 