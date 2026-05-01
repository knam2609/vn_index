import os
import sys
from sklearn.preprocessing import StandardScaler
from scripts.s3_scripts.read_write_to_s3 import read_csv_from_s3, write_df_to_s3

# Add parent directory for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
# PyTorch 2.6+ compatibility for checkpoints loaded by NeuralProphet/Lightning.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from scripts.general_scripts.forecast_run import DailyForecastConfig, run_daily_forecast
from scripts.general_scripts.helper import CustomizedLoss

# Constants
CONFIG = DailyForecastConfig()
INPUT_DIR = "ready_data/vn_index_data"
OUTPUT_DIR = "forecast_result/vn_index"

def main():
    # Ensure output directory exists
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load cleaned data
    data_path = os.path.join(INPUT_DIR, "cleaned_vn_index_data.csv")
    df = read_csv_from_s3("vn-index", data_path)

    failed_models = []

    for model_name in CONFIG.model_names:
        print(f"🔄 Running daily forecast for {model_name}...")

        try:
            artifacts = run_daily_forecast(
                df,
                model_name,
                config=CONFIG,
                scaler=StandardScaler(),
                criterion=CustomizedLoss(),
            )

            # Define file paths
            forecast_path = os.path.join(OUTPUT_DIR, f"forecast_{model_name}.csv")
            metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{model_name}.csv")
            final_path = os.path.join(OUTPUT_DIR, f"final_{model_name}.csv")

            # Save all outputs
            write_df_to_s3(artifacts.forecast_df, "vn-index", forecast_path)
            write_df_to_s3(artifacts.metrics_df, "vn-index", metrics_path)
            write_df_to_s3(artifacts.final_df, "vn-index", final_path)

            print(f"✅ Saved forecast to {forecast_path}")
            print(f"✅ Saved metrics to {metrics_path}")
            print(f"✅ Saved final comparison to {final_path}")

        except Exception as e:
            print(f"❌ Failed to run {model_name}: {e}")
            failed_models.append(model_name)

    if failed_models:
        raise RuntimeError(
            f"Model forecasting failed for: {', '.join(failed_models)}"
        )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
