from .features_engineering import quicky_data, lag_features_indicators
from .pipelines import price_model
from .predict import future_price_prediction
from .lstm import LSTMModelMultiOutput
from .transformer import TimeSeriesTransformerMultiOutput
from .helper import CustomizedLoss
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from neuralprophet import NeuralProphet
from .neural_prophet import neural_prophet_model
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

def directional_accuracy_general(true_series, pred_series, n_forecasts):
    """
    Computes directional accuracy:
    - For n_forecasts = 1: pred(i) - true(i-1) vs true(i) - true(i-1)
    - For n_forecasts > 1:
        First step: pred(i) - true(i-1) vs true(i) - true(i-1)
        Remaining steps: pred(j) - pred(j-1) vs true(j) - true(j-1)
    """
    correct_total = 0
    total_points = 0

    if n_forecasts == 1:
        for i in range(1, len(true_series)):
            pred_change = pred_series.iloc[i] - true_series.iloc[i - 1]
            true_change = true_series.iloc[i] - true_series.iloc[i - 1]
            if np.sign(pred_change) == np.sign(true_change):
                correct_total += 1
            total_points += 1

    else:
        for i in range(1, len(true_series) - n_forecasts + 1, n_forecasts):
            anchor_idx = i - 1
            if anchor_idx < 0:
                continue  # skip if there's no anchor true(i-1)

            true_block = true_series.iloc[i:i + n_forecasts].reset_index(drop=True)
            pred_block = pred_series.iloc[i:i + n_forecasts].reset_index(drop=True)

            if len(true_block) < n_forecasts or len(pred_block) < n_forecasts:
                continue  # skip incomplete blocks

            correct = 0

            # First step: pred(i) - true(i-1)
            true_change = true_block.iloc[0] - true_series.iloc[anchor_idx]
            pred_change = pred_block.iloc[0] - true_series.iloc[anchor_idx]
            if np.sign(true_change) == np.sign(pred_change):
                correct += 1

            # Remaining steps: pred(j) - pred(j-1) vs true(j) - true(j-1)
            for j in range(1, n_forecasts):
                true_step = true_block.iloc[j] - true_block.iloc[j - 1]
                pred_step = pred_block.iloc[j] - pred_block.iloc[j - 1]
                if np.sign(true_step) == np.sign(pred_step):
                    correct += 1

            correct_total += correct
            total_points += n_forecasts

    return correct_total / total_points if total_points > 0 else np.nan

def test_predict(df, n_tests, n_forecasts, seasonal_periods, scaler, model_type, criterion, n_lags):
    # TEST
    test_df = df[['Date', 'VN_Index_Close']].iloc[[-n_tests-1]]
    base_df = df[['Date', 'VN_Index_Close']].iloc[[-n_tests-1]]
    true_df = df[['Date', 'VN_Index_Close']].iloc[-n_tests-1:]
    test_df.rename(columns={"VN_Index_Close": "Predicted VN-INDEX"}, inplace=True)
    base_df.rename(columns={"VN_Index_Close": "Baseline Predicted VN-INDEX"}, inplace=True)
    true_df.rename(columns={"VN_Index_Close": "Actual VN-INDEX"}, inplace=True)
    print(base_df)
    print(test_df)

    for i in range(n_tests//n_forecasts):
        df_2 = df.copy().iloc[:-n_forecasts*(i+1)]
        series = df_2['VN_Index_Close'].copy()
        # Suppress only statsmodels warnings during model fitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ValueWarning)
            warnings.simplefilter("ignore", category=FutureWarning)

            # Base model
            baseline = ExponentialSmoothing(
            series,
            trend='add',            # additive trend
            seasonal='add',         # additive seasonality
            seasonal_periods=seasonal_periods
            ).fit()

            # 2) Forecast the next day (one‐step ahead):
            price_forecast = baseline.forecast(n_forecasts)

        # ✅ Get the actual future dates from df
        forecast_start_idx = len(df) - n_forecasts * (i + 1)
        forecast_dates = df['Date'].iloc[forecast_start_idx:forecast_start_idx + n_forecasts].reset_index(drop=True)

        baseline_df = pd.DataFrame({
            'Date': forecast_dates,
            'Baseline Predicted VN-INDEX': price_forecast.values,
        })

        print(baseline_df)

        # Store base model forecast
        base_df = pd.concat([base_df, baseline_df], ignore_index=True)

        if model_type == 'NeuralProphet':
            result_df = neural_prophet_model(df_2, n_lags, n_forecasts)

        else:
            if model_type == 'LSTM':
                algo = LSTMModelMultiOutput
            else:
                algo = TimeSeriesTransformerMultiOutput
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ValueWarning)
                warnings.simplefilter("ignore", category=FutureWarning)
                df_2 = quicky_data(df_2, seasonal_periods)
            
            data = df_2[['residual', 'trend', 'seasonal', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']]
            
            # 🚀 Train the model and get the test set
            model, X_test_tensor, scaler, y_pred = price_model(data, scaler, algo, criterion, tuning=False, train_seq_len=n_lags, test_seq_len=n_forecasts, seasonal_periods=seasonal_periods, epochs=50, verbose=False) # type: ignore
            result_df = future_price_prediction(X_test_tensor, data, y_pred, scaler, model, num_days=n_forecasts, seasonal_periods=seasonal_periods, verbose=False)
            print(result_df)

        test_df = pd.concat([test_df, result_df], ignore_index=True)

    true_df['Date'] = pd.to_datetime(true_df['Date'])
    base_df['Date'] = pd.to_datetime(base_df['Date'])
    test_df['Date'] = pd.to_datetime(test_df['Date'])

    final_df = pd.merge(true_df, pd.merge(base_df, test_df, on='Date', how='inner'), on='Date', how='inner')
    actual = final_df['Actual VN-INDEX']
    base = final_df['Baseline Predicted VN-INDEX']
    predict = final_df['Predicted VN-INDEX']
    
    metrics_df = pd.DataFrame({
        'Model': ['ExponentialSmoothing', model_type],
        'MAE': [
            mean_absolute_error(actual.iloc[-n_tests:], base.iloc[-n_tests:]),
            mean_absolute_error(actual.iloc[-n_tests:], predict.iloc[-n_tests:])
        ],
        'RMSE': [
            np.sqrt(mean_squared_error(actual.iloc[-n_tests:], base.iloc[-n_tests:])),
            np.sqrt(mean_squared_error(actual.iloc[-n_tests:], predict.iloc[-n_tests:]))
        ],
        'Directional Accuracy': [
            directional_accuracy_general(actual, base, n_forecasts),
            directional_accuracy_general(actual, predict, n_forecasts)
        ]
    })

    # FORECAST
    if model_type == 'NeuralProphet':
        forecast_df = neural_prophet_model(df, n_lags, n_forecasts)

    else:
        if model_type == 'LSTM':
            algo = LSTMModelMultiOutput
        else:
            algo = TimeSeriesTransformerMultiOutput

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ValueWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            df_1 = quicky_data(df.copy(), seasonal_periods)
        data = df_1[['residual', 'trend', 'seasonal', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']]
        
        # 🚀 Train the model and get the test set
        model, X_test_tensor, scaler, y_pred = price_model(data, scaler, algo, criterion, tuning=False, train_seq_len=n_lags, test_seq_len=n_forecasts, seasonal_periods=seasonal_periods, epochs=50, verbose=False) # type: ignore
        forecast_df = future_price_prediction(X_test_tensor, data, y_pred, scaler, model, num_days=n_forecasts, seasonal_periods=seasonal_periods, verbose=False)
        forecast_df.rename(columns={"Predicted VN-INDEX": "Future Predictions"}, inplace=True)

    return final_df.iloc[-n_tests:], metrics_df, forecast_df



    

        