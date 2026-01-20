#!/bin/bash
set -euo pipefail

# Resolve repo root no matter where this script is called from
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Make both of these import styles work:
#  - from scripts.s3_scripts...
#  - from s3_scripts...
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/scripts"

echo "PYTHONPATH=$PYTHONPATH"
python -V

# Step 1: Scrape the latest VN-Index data
python -m scripts.vn_index_scripts.scrape_vn_index || echo "Scraping failed"

# Step 2: Process the raw data
python -m scripts.vn_index_scripts.vn_index_preprocessing || echo "Processing failed"

# Step 3: Run all model forecasts and save output
python -m render_app.run_test_predict || echo "Forecasting failed"