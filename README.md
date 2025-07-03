# 🇻🇳 VN-Index Forecasting Dashboard
This project benchmarks modern time series models for short-term forecasting of the VN-Index, Vietnam’s major stock market index. It is built to support investors, analysts, and financial platforms with daily updated forecasts and a live interactive dashboard.

## 🧠 Benchmark Models
We evaluate the performance of the following deep learning models:

LSTM – Sequence learning using memory cells.

Transformer – Attention-based model adapted from NLP for financial forecasting.

NeuralProphet – Facebook Prophet extended with deep learning capabilities.

Exponential Smoothing – Strong classical baseline for time series.

Each model is assessed on:

MAE / RMSE – Accuracy of price level.

Directional Accuracy – Correctly predicting upward/downward movements.

## 🧪 Daily Forecast Automation
To update forecasts and keep the dashboard current: bash render_app/run_scrape.sh

This script:

Scrapes VN-Index data from StockBiz.

Process the data.

Precomputes forecasts for all models and saves to CSV.

## 📊 Live Dashboard
A Streamlit-powered dashboard presents:

Test set prediction charts

2-day VN-Index forecast

Model comparison table

🔗 View it live: https://vn-index.onrender.com