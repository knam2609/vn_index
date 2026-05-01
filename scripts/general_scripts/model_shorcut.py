from .forecast_run import DailyForecastConfig, directional_accuracy_general, run_daily_forecast


def test_predict(df, n_tests, n_forecasts, seasonal_periods, scaler, model_type, criterion, n_lags):
    config = DailyForecastConfig(
        n_tests=n_tests,
        n_forecasts=n_forecasts,
        seasonal_periods=seasonal_periods,
        n_lags=n_lags,
    )
    artifacts = run_daily_forecast(
        df,
        model_type,
        config=config,
        scaler=scaler,
        criterion=criterion,
    )
    return artifacts.as_tuple()
