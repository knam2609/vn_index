import numpy as np
import pandas as pd
import pytest

from scripts.general_scripts.forecast_run import (
    DailyForecastConfig,
    ForecastArtifacts,
    directional_accuracy_general,
    run_all_daily_forecasts,
    run_daily_forecast,
)


class FakeForecastAdapter:
    model_name = "FakeModel"

    def __init__(self):
        self.calls = []

    def forecast(self, df, *, config, scaler, criterion):
        training_df = df.copy()
        self.calls.append(training_df)
        future_dates = pd.bdate_range(
            training_df["Date"].iloc[-1] + pd.offsets.BDay(1),
            periods=config.n_forecasts,
        )
        last_close = float(training_df["VN_Index_Close"].iloc[-1])
        predictions = [
            last_close + step
            for step in range(1, config.n_forecasts + 1)
        ]
        return pd.DataFrame(
            {
                "Date": future_dates,
                "Predicted VN-INDEX": predictions,
            }
        )


def make_vn_index_df(rows=8):
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2026-01-01", periods=rows),
            "VN_Index_Close": [100 + i for i in range(rows)],
        }
    )


def baseline_offset_forecast(series, n_forecasts, config):
    return np.array(
        [
            float(series.iloc[-1]) + 2 * step
            for step in range(1, n_forecasts + 1)
        ]
    )


def small_config():
    return DailyForecastConfig(
        n_tests=4,
        n_forecasts=2,
        seasonal_periods=2,
        n_lags=2,
        epochs=1,
        model_names=("FakeModel",),
    )


def test_run_daily_forecast_returns_dashboard_artifacts_with_fake_adapter():
    adapter = FakeForecastAdapter()
    artifacts = run_daily_forecast(
        make_vn_index_df(),
        "FakeModel",
        config=small_config(),
        scaler=object(),
        criterion=object(),
        adapters={"FakeModel": adapter},
        baseline_forecaster=baseline_offset_forecast,
    )

    assert list(artifacts.final_df.columns) == [
        "Date",
        "Actual VN-INDEX",
        "Baseline Predicted VN-INDEX",
        "Predicted VN-INDEX",
    ]
    assert list(artifacts.metrics_df.columns) == [
        "Model",
        "MAE",
        "RMSE",
        "Directional Accuracy",
    ]
    assert list(artifacts.forecast_df.columns) == ["Date", "Future Predictions"]
    assert len(artifacts.final_df) == 4
    assert len(artifacts.forecast_df) == 2

    assert [len(call) for call in adapter.calls] == [6, 4, 8]
    assert artifacts.forecast_df["Future Predictions"].tolist() == [108.0, 109.0]

    model_metrics = artifacts.metrics_df[
        artifacts.metrics_df["Model"] == "FakeModel"
    ].iloc[0]
    assert model_metrics["MAE"] == pytest.approx(0.0)
    assert model_metrics["RMSE"] == pytest.approx(0.0)
    assert model_metrics["Directional Accuracy"] == pytest.approx(1.0)


def test_run_daily_forecast_validates_missing_columns():
    with pytest.raises(ValueError, match="Missing required VN-Index columns"):
        run_daily_forecast(
            pd.DataFrame({"Date": pd.bdate_range("2026-01-01", periods=8)}),
            "FakeModel",
            config=small_config(),
        )


def test_run_daily_forecast_validates_insufficient_rows():
    with pytest.raises(ValueError, match="Insufficient VN-Index rows"):
        run_daily_forecast(
            make_vn_index_df(rows=5),
            "FakeModel",
            config=small_config(),
        )


def test_run_all_daily_forecasts_uses_configured_models():
    adapter = FakeForecastAdapter()
    results = run_all_daily_forecasts(
        make_vn_index_df(),
        config=small_config(),
        adapters={"FakeModel": adapter},
        scaler_factory=object,
        criterion_factory=object,
        baseline_forecaster=baseline_offset_forecast,
    )

    assert list(results) == ["FakeModel"]
    assert isinstance(results["FakeModel"], ForecastArtifacts)
    assert [len(call) for call in adapter.calls] == [6, 4, 8]


def test_directional_accuracy_general_for_single_step_changes():
    actual = pd.Series([10, 12, 11])
    predicted = pd.Series([10, 9, 8])

    assert directional_accuracy_general(actual, predicted, n_forecasts=1) == pytest.approx(0.5)


def test_legacy_test_predict_returns_existing_tuple(monkeypatch):
    from scripts.general_scripts import model_shorcut

    expected = ForecastArtifacts(
        final_df=pd.DataFrame({"Date": []}),
        metrics_df=pd.DataFrame({"Model": []}),
        forecast_df=pd.DataFrame({"Date": []}),
    )
    seen = {}

    def fake_run_daily_forecast(df, model_name, *, config, scaler, criterion):
        seen["df"] = df
        seen["model_name"] = model_name
        seen["config"] = config
        seen["scaler"] = scaler
        seen["criterion"] = criterion
        return expected

    monkeypatch.setattr(model_shorcut, "run_daily_forecast", fake_run_daily_forecast)

    df = make_vn_index_df()
    scaler = object()
    criterion = object()
    final_df, metrics_df, forecast_df = model_shorcut.test_predict(
        df,
        n_tests=4,
        n_forecasts=2,
        seasonal_periods=2,
        scaler=scaler,
        model_type="FakeModel",
        criterion=criterion,
        n_lags=2,
    )

    assert final_df is expected.final_df
    assert metrics_df is expected.metrics_df
    assert forecast_df is expected.forecast_df
    assert seen["df"] is df
    assert seen["model_name"] == "FakeModel"
    assert seen["config"].n_tests == 4
    assert seen["config"].n_forecasts == 2
    assert seen["config"].seasonal_periods == 2
    assert seen["config"].n_lags == 2
    assert seen["scaler"] is scaler
    assert seen["criterion"] is criterion
