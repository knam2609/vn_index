# -----------------------------
# Future Prediction Function
# -----------------------------
import torch
import pandas as pd
import matplotlib.pyplot as plt
from .helper import inverse_scale_predictions, device
from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing

def future_change_prediction(X_test, data, pred_close, scaler, model, num_days=10, df=None):
    """
    Generate future predictions for VN-INDEX, now including cyclical
    day‑of‑week and month embeddings at each step.
      - X_test:      Tensor of shape (n_samples, seq_len, n_features) or (1, seq_len, n_features)
      - data:        original DataFrame (with DateTimeIndex)
      - pred_close:  list or array of past test‑set predictions (optional)
      - scaler:      StandardScaler or MinMaxScaler fitted on all features
      - model:       your trained LSTM/self‑attn model
      - num_days:    number of future trading days to predict
      - df:          DataFrame with 'VN_Index_Close' (if separate from data)
    """
    source_df = df if df is not None else data
    if isinstance(data, pd.Series):
        data = data.to_frame()

    last_close = float(source_df["VN_Index_Close"].iloc[-1])
    last_date  = source_df.index[-1]

    future_dates     = []
    future_close     = []

    # 1) predict next‑day % change (scaled)
    with torch.no_grad():
            pred_scaled = model(X_test.unsqueeze(0))

    print(pred_scaled.shape)
    # inverse‐scale
    future_preds_pct = inverse_scale_predictions(pred_scaled, scaler).detach().cpu().numpy().ravel()

    for _ in range(num_days):
        # next trading date
        next_date = last_date + pd.Timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += pd.Timedelta(days=1)
        future_dates.append(next_date)

        # compute predicted price
        new_close = last_close * (1 + future_preds_pct[_])
        future_close.append(new_close)

        # update for next step
        last_close = new_close
        last_date  = next_date

    # 8) Plot
    plt.figure(figsize=(12, 6))
    hist_dates  = source_df.index[-num_days:]
    hist_values = source_df.loc[hist_dates, "VN_Index_Close"].values
    plt.plot(hist_dates, hist_values, label="Historical VN-INDEX", color="blue")

    if pred_close is not None:
        test_plot_dates = hist_dates[-len(pred_close):]
        plt.plot(test_plot_dates, pred_close, label="Test Predictions", color="red")

    plt.plot(future_dates, future_close,
             marker="o", linestyle="--", color="green",
             label="Future Predictions")
    plt.xlabel("Date")
    plt.ylabel("VN-INDEX")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.title("Historical, Test & Future VN-INDEX")
    plt.show()

    # 9) Return DataFrame
    future_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted VN-INDEX": future_close,
        "Predicted Change":   future_preds_pct
    })
    print(future_df)

# Future Prediction Function
def future_price_prediction(X_test, data, y_pred, scaler, model, num_days=10, seasonal_periods=261, verbose=True):
    if isinstance(data, pd.Series):
        data = data.to_frame()

    model.eval()

    last_date = data.index[-1]
    future_dates = []
    # compute next trading date
    for i in range(num_days):
        next_date = last_date + pd.Timedelta(days=1)
        while next_date.weekday() >= 5:  # skip weekends
            next_date += pd.Timedelta(days=1)
        future_dates.append(next_date)
        last_date  = next_date

    with torch.no_grad():
            pred_scaled = model(X_test.unsqueeze(0))

    seasonal = data['seasonal']
    seasonal_last = seasonal.iloc[-num_days:]
    # ets = ExponentialSmoothing(
    #     seasonal,
    #     seasonal='add',
    #     seasonal_periods=seasonal_periods
    # ).fit()
    # seasonal_next = ets.forecast(num_days)
    from scipy.ndimage import uniform_filter1d
    smoothed_seasonal = uniform_filter1d(seasonal.values, size=7)
    seasonal_next = smoothed_seasonal[-num_days:]
    if verbose:
        print(seasonal_next)
    
    trend = data['trend']
    trend_model = Holt(trend).fit()
    trend_last = trend.iloc[-num_days:]    
    trend_next  = trend_model.forecast(num_days)
    if verbose:
        print(trend_next)

    # inverse‐scale
    future_preds = inverse_scale_predictions(pred_scaled, scaler).detach().cpu().numpy().ravel() + seasonal_next + trend_next
    if verbose:
        plt.figure(figsize=(12,6))
        plt.plot(future_dates, future_preds, marker='o', linestyle="dashed", color="red", label="Predicted VN-INDEX")
        plt.xlabel("Date")
        plt.ylabel("VN-INDEX")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.title(f"Predicted VN-INDEX for Next {num_days} Trading Days")
        plt.show()

        historical_dates = data.index[-num_days:]
        historical_values = data.iloc[-num_days:, 0].values + seasonal_last + trend_last
        plt.figure(figsize=(12,6))
        plt.plot(historical_dates, historical_values, label="Historical VN-INDEX", color="blue")
        plt.plot(historical_dates, y_pred, label="Test Predictions", color="red")
        plt.plot(future_dates, future_preds, color="green", label="Future Predicted VN-INDEX")
        plt.xlabel("Date")
        plt.ylabel("VN-INDEX")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.title("Historical VN-INDEX with Future Predictions")
        plt.show()

    future_df = pd.DataFrame({"Date": future_dates, "Predicted VN-INDEX": future_preds})
    return future_df