import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 🧭 Page setup
st.set_page_config(page_title="VN-Index Forecasting", layout="wide")
st.title("📈 VN-Index Forecasting Dashboard (From Saved Forecasts)")

# ⚙️ User input
model_choice = st.selectbox("Choose Forecasting Model:", ["LSTM", "Transformer", "NeuralProphet"])

# 📦 File paths
BASE_DIR = "forecast_result/vn_index"
forecast_path = os.path.join(BASE_DIR, f"forecast_{model_choice}.csv")
metrics_path = os.path.join(BASE_DIR, f"metrics_{model_choice}.csv")
final_path = os.path.join(BASE_DIR, f"final_{model_choice}.csv")

# 📦 Load saved CSVs
try:
    forecast_df = pd.read_csv(forecast_path, parse_dates=["Date"])
    metrics_df = pd.read_csv(metrics_path)
    final_df = pd.read_csv(final_path, parse_dates=["Date"])

    # ✅ Check loaded data
    if final_df.empty or forecast_df.empty or metrics_df.empty:
        st.warning("⚠️ One or more required data files are empty.")
    else:
        # 📊 Plot actual vs predicted (from test set)
        st.subheader("📊 Actual vs Predicted VN-Index (Test Set)")
        fig, ax = plt.subplots()
        ax.plot(final_df["Date"], final_df["Actual VN-INDEX"], label="Actual")
        ax.plot(final_df["Date"], final_df["Predicted VN-INDEX"], label="Predicted")
        ax.set_xlabel("Date")
        ax.set_ylabel("VN-Index")
        ax.legend()

        # ✅ Tilt x-axis labels for better readability
        plt.xticks(rotation=60, ha='right')
        plt.tight_layout()

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))

        st.pyplot(fig)

        # 📋 Metrics
        st.subheader("📌 Model Performance")
        st.dataframe(metrics_df)

        # 📑 Forecast Table
        st.subheader("🔍 Forecast Table (Next 2 Days)")
        st.dataframe(forecast_df.tail(10))

except FileNotFoundError:
    st.error(f"❌ Missing forecast, final, or metrics CSV for model: {model_choice}")