import streamlit as st
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
from sklearn.preprocessing import StandardScaler

# ✅ Add parent directory to Python path so Render can find 'scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.general_scripts.helper import CustomizedLoss
from scripts.general_scripts.model_shorcut import test_predict

# 🧭 Page setup
st.set_page_config(page_title="VN-Index Forecasting", layout="wide")
st.title("📈 VN-Index Forecasting Dashboard (Real-time)")

# ⚙️ User input
model_choice = st.selectbox("Choose Forecasting Model:", ["LSTM", "Transformer", "NeuralProphet"])
forecast_days = st.slider("Number of Days to Forecast:", min_value=1, max_value=2, value=1)
test_days = st.slider("Number of Days to Test:", min_value=forecast_days, max_value=20, value=forecast_days)

# 🔄 Refresh & run scraping scripts if needed
if st.button("🔄 Refresh Forecast"):
    with st.spinner("Scraping and processing new data..."):
        subprocess.run(["python", "scripts/vn_index_scripts/scrape_vn_index.py"])
        subprocess.run(["python", "scripts/vn_index_scripts/vn_index_processing.py"])
        st.session_state.update_trigger = True

# 🧠 Caching
if 'update_trigger' not in st.session_state:
    st.session_state.update_trigger = False

if st.session_state.update_trigger:
    st.cache_data.clear()
    st.session_state.update_trigger = False

# 📦 Load processed data
@st.cache_data(ttl=3600)
def load_data():
    cleaned_path = "ready_data/vn_index_data/cleaned_vn_index_data.csv"
    try:
        return pd.read_csv(cleaned_path, parse_dates=["Date"])
    except FileNotFoundError:
        subprocess.run(["python", "scripts/vn_index_scripts/scrape_vn_index.py"])
        subprocess.run(["python", "scripts/vn_index_scripts/vn_index_processing.py"])
        return pd.read_csv(cleaned_path, parse_dates=["Date"])

df = load_data()

# 🔮 Run prediction
try:
    final_df, metrics_df, forecast_df = test_predict(
        df=df,
        n_tests=test_days,
        n_forecasts=forecast_days,
        seasonal_periods=261,
        scaler=StandardScaler(),
        model_type=model_choice,
        criterion=CustomizedLoss(),
        n_lags=5
    )

    # 📊 Plot forecast
    st.subheader("📊 Actual vs Predicted VN-Index")
    fig, ax = plt.subplots()
    ax.plot(forecast_df["Date"], forecast_df["Actual VN-INDEX"], label="Actual")
    ax.plot(forecast_df["Date"], forecast_df["Predicted VN-INDEX"], label="Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("VN-Index")
    ax.legend()
    st.pyplot(fig)

    # 📋 Metrics
    st.subheader("📌 Model Performance")
    st.dataframe(metrics_df)

    # 📑 Forecast Table
    st.subheader("🔍 Forecast Table")
    st.dataframe(forecast_df.tail(10))

except Exception as e:
    st.error(f"⚠️ Forecasting failed: {e}")
