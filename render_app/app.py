import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 🧭 Page setup
st.set_page_config(page_title="VN-Index Forecasting", layout="wide")
st.title("📈 VN-Index Forecasting Dashboard")

# ⚙️ User input
model_choice = st.selectbox("Choose Forecasting Model:", ["LSTM", "Transformer", "NeuralProphet"])

# 📦 File paths
BASE_DIR = "forecast_result/vn_index"
forecast_path = os.path.join(BASE_DIR, f"forecast_{model_choice}.csv")
metrics_path = os.path.join(BASE_DIR, f"metrics_{model_choice}.csv")
final_path = os.path.join(BASE_DIR, f"final_{model_choice}.csv")

def highlight_changes_with_base(s, base_value):
    colors = []
    prev = base_value
    for val in s:
        if val > prev:
            colors.append('color: green')
        elif val < prev:
            colors.append('color: red')
        else:
            colors.append('')
        prev = val
    return colors


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

        # 📑 Forecast Table (Next 2 Days)
        st.subheader("🔍 Forecast Table (Next 2 Days)")

        recent_forecast = forecast_df

        # Ensure required column exists
        target_col = 'Future Predictions'
        if target_col in recent_forecast.columns and 'Actual VN-INDEX' in final_df.columns:
            # Get the last actual value from final_df
            last_actual_value = final_df['Actual VN-INDEX'].iloc[-1]

            # Apply conditional formatting using the last historical value as starting point
            styled_df = recent_forecast.style.apply(
                highlight_changes_with_base, 
                base_value=last_actual_value,
                subset=[target_col]
            )
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.dataframe(recent_forecast, use_container_width=True)


except FileNotFoundError:
    st.error(f"❌ Missing forecast, final, or metrics CSV for model: {model_choice}")