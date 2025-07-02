#!/bin/bash

# Step 1: Scrape the latest VN-Index data
python scripts/vn_index_scripts/scrape_vn_index.py || echo "Scraping failed"

# Step 2: Process the raw data
python scripts/vn_index_scripts/vn_index_preprocessing.py || echo "Processing failed"

# Step 3: Run all model forecasts and save output
python render_app/run_test_predict.py || echo "Forecasting failed"
