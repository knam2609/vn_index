import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scripts.s3_scripts.read_write_to_s3 import read_csv_from_s3, write_df_to_s3

# Add parent directory for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import contextlib
import warnings
# Completely suppress all warnings
def custom_suppress_warning(*args, **kwargs):
    pass

warnings.showwarning = custom_suppress_warning
warnings.filterwarnings("ignore")
import logging

# Suppress all logging
logging.basicConfig(level=logging.CRITICAL + 1)
# Force matplotlib backend to avoid plotly attempts
os.environ["NP_PLOT_BACKEND"] = "matplotlib"

# Suppress stderr where the print happens
@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Wrap import
with suppress_stdout_stderr():
    from scripts.general_scripts.model_shorcut import test_predict
from scripts.general_scripts.helper import CustomizedLoss

# Constants
N_FORECASTS = 2
N_TESTS = 20
SEASONAL_PERIODS = 261
N_LAGS = 5
INPUT_DIR = "ready_data/vn_index_data"
OUTPUT_DIR = "forecast_result/vn_index"

def main():
    # Ensure output directory exists
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load cleaned data
    data_path = os.path.join(INPUT_DIR, "cleaned_vn_index_data.csv")
    df = read_csv_from_s3("vn-index", data_path)

    # Model list
    models = ["LSTM", "Transformer", "NeuralProphet"]

    for model_name in models:
        print(f"🔄 Running test_predict for {model_name}...")

        try:
            final_df, metrics_df, forecast_df = test_predict(
                df=df,
                n_tests=N_TESTS,
                n_forecasts=N_FORECASTS,
                seasonal_periods=SEASONAL_PERIODS,
                scaler=StandardScaler(),
                model_type=model_name,
                criterion=CustomizedLoss(),
                n_lags=N_LAGS,
            )

            # Define file paths
            forecast_path = os.path.join(OUTPUT_DIR, f"forecast_{model_name}.csv")
            metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{model_name}.csv")
            final_path = os.path.join(OUTPUT_DIR, f"final_{model_name}.csv")

            # Save all outputs
            write_df_to_s3(forecast_df, "vn-index", forecast_path)
            write_df_to_s3(metrics_df, "vn-index", metrics_path)
            write_df_to_s3(final_df, "vn-index", final_path)

            print(f"✅ Saved forecast to {forecast_path}")
            print(f"✅ Saved metrics to {metrics_path}")
            print(f"✅ Saved final comparison to {final_path}")

        except Exception as e:
            print(f"❌ Failed to run {model_name}: {e}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
