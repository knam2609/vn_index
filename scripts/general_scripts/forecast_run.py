from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = ("Date", "VN_Index_Close")
PREDICTION_COLUMNS = ("Date", "Predicted VN-INDEX")
FEATURE_COLUMNS = (
    "residual",
    "trend",
    "seasonal",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
)


@dataclass(frozen=True)
class DailyForecastConfig:
    n_tests: int = 20
    n_forecasts: int = 2
    seasonal_periods: int = 261
    n_lags: int = 5
    epochs: int = 50
    model_names: tuple[str, ...] = ("LSTM", "Transformer", "NeuralProphet")


@dataclass(frozen=True)
class ForecastArtifacts:
    final_df: pd.DataFrame
    metrics_df: pd.DataFrame
    forecast_df: pd.DataFrame

    def as_tuple(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.final_df, self.metrics_df, self.forecast_df


class ForecastModelAdapter(Protocol):
    model_name: str

    def forecast(
        self,
        df: pd.DataFrame,
        *,
        config: DailyForecastConfig,
        scaler,
        criterion,
    ) -> pd.DataFrame:
        """Return Date and Predicted VN-INDEX columns for the next horizon."""


BaselineForecaster = Callable[
    [pd.Series, int, DailyForecastConfig],
    Sequence[float] | np.ndarray | pd.Series,
]


@dataclass(frozen=True)
class TorchForecastModelAdapter:
    model_name: str
    model_type: type

    def forecast(
        self,
        df: pd.DataFrame,
        *,
        config: DailyForecastConfig,
        scaler,
        criterion,
    ) -> pd.DataFrame:
        from .features_engineering import quicky_data
        from .pipelines import price_model
        from .predict import future_price_prediction

        with _suppress_model_warnings():
            feature_df = quicky_data(df.copy(), config.seasonal_periods)

        data = feature_df[list(FEATURE_COLUMNS)]
        model, x_test_tensor, scaler, y_pred = price_model(
            data,
            scaler,
            self.model_type,
            criterion,
            tuning=False,
            train_seq_len=config.n_lags,
            test_seq_len=config.n_forecasts,
            seasonal_periods=config.seasonal_periods,
            epochs=config.epochs,
            verbose=False,
        )
        return future_price_prediction(
            x_test_tensor,
            data,
            y_pred,
            scaler,
            model,
            num_days=config.n_forecasts,
            seasonal_periods=config.seasonal_periods,
            verbose=False,
        )


@dataclass(frozen=True)
class NeuralProphetForecastModelAdapter:
    model_name: str = "NeuralProphet"

    def forecast(
        self,
        df: pd.DataFrame,
        *,
        config: DailyForecastConfig,
        scaler,
        criterion,
    ) -> pd.DataFrame:
        from .neural_prophet import neural_prophet_model

        return neural_prophet_model(df, config.n_lags, config.n_forecasts)


def create_default_model_adapters() -> dict[str, ForecastModelAdapter]:
    from .lstm import LSTMModelMultiOutput
    from .transformer import TimeSeriesTransformerMultiOutput

    return {
        "LSTM": TorchForecastModelAdapter("LSTM", LSTMModelMultiOutput),
        "Transformer": TorchForecastModelAdapter(
            "Transformer",
            TimeSeriesTransformerMultiOutput,
        ),
        "NeuralProphet": NeuralProphetForecastModelAdapter(),
    }


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
                continue

            true_block = true_series.iloc[i:i + n_forecasts].reset_index(drop=True)
            pred_block = pred_series.iloc[i:i + n_forecasts].reset_index(drop=True)

            if len(true_block) < n_forecasts or len(pred_block) < n_forecasts:
                continue

            correct = 0

            true_change = true_block.iloc[0] - true_series.iloc[anchor_idx]
            pred_change = pred_block.iloc[0] - true_series.iloc[anchor_idx]
            if np.sign(true_change) == np.sign(pred_change):
                correct += 1

            for j in range(1, n_forecasts):
                true_step = true_block.iloc[j] - true_block.iloc[j - 1]
                pred_step = pred_block.iloc[j] - pred_block.iloc[j - 1]
                if np.sign(true_step) == np.sign(pred_step):
                    correct += 1

            correct_total += correct
            total_points += n_forecasts

    return correct_total / total_points if total_points > 0 else np.nan


def run_daily_forecast(
    df: pd.DataFrame,
    model_name: str,
    *,
    config: DailyForecastConfig | None = None,
    scaler=None,
    criterion=None,
    adapters: Mapping[str, ForecastModelAdapter] | None = None,
    baseline_forecaster: BaselineForecaster | None = None,
) -> ForecastArtifacts:
    config = config or DailyForecastConfig()
    _validate_config(config)

    data = _prepare_vn_index_data(df, config)
    adapter = _resolve_adapter(model_name, adapters)
    scaler = scaler if scaler is not None else StandardScaler()
    criterion = criterion if criterion is not None else _default_criterion()
    baseline_forecaster = baseline_forecaster or _exponential_smoothing_forecast

    true_df = data[["Date", "VN_Index_Close"]].iloc[-config.n_tests - 1:].copy()
    true_df.rename(columns={"VN_Index_Close": "Actual VN-INDEX"}, inplace=True)

    base_df = data[["Date", "VN_Index_Close"]].iloc[[-config.n_tests - 1]].copy()
    base_df.rename(columns={"VN_Index_Close": "Baseline Predicted VN-INDEX"}, inplace=True)

    test_df = data[["Date", "VN_Index_Close"]].iloc[[-config.n_tests - 1]].copy()
    test_df.rename(columns={"VN_Index_Close": "Predicted VN-INDEX"}, inplace=True)

    for i in range(config.n_tests // config.n_forecasts):
        training_df = data.copy().iloc[:-config.n_forecasts * (i + 1)]
        baseline_values = baseline_forecaster(
            training_df["VN_Index_Close"].copy(),
            config.n_forecasts,
            config,
        )
        forecast_start_idx = len(data) - config.n_forecasts * (i + 1)
        forecast_dates = data["Date"].iloc[
            forecast_start_idx:forecast_start_idx + config.n_forecasts
        ].reset_index(drop=True)

        baseline_df = pd.DataFrame(
            {
                "Date": forecast_dates,
                "Baseline Predicted VN-INDEX": np.asarray(baseline_values),
            }
        )
        base_df = pd.concat([base_df, baseline_df], ignore_index=True)

        result_df = adapter.forecast(
            training_df,
            config=config,
            scaler=scaler,
            criterion=criterion,
        )
        result_df = _normalize_prediction_frame(
            result_df,
            expected_horizon=config.n_forecasts,
            model_name=model_name,
        )
        test_df = pd.concat([test_df, result_df], ignore_index=True)

    true_df["Date"] = pd.to_datetime(true_df["Date"])
    base_df["Date"] = pd.to_datetime(base_df["Date"])
    test_df["Date"] = pd.to_datetime(test_df["Date"])

    final_df = pd.merge(
        true_df,
        pd.merge(base_df, test_df, on="Date", how="inner"),
        on="Date",
        how="inner",
    )
    actual = final_df["Actual VN-INDEX"]
    base = final_df["Baseline Predicted VN-INDEX"]
    predict = final_df["Predicted VN-INDEX"]

    metrics_df = pd.DataFrame(
        {
            "Model": ["ExponentialSmoothing", model_name],
            "MAE": [
                mean_absolute_error(actual.iloc[-config.n_tests:], base.iloc[-config.n_tests:]),
                mean_absolute_error(actual.iloc[-config.n_tests:], predict.iloc[-config.n_tests:]),
            ],
            "RMSE": [
                np.sqrt(mean_squared_error(actual.iloc[-config.n_tests:], base.iloc[-config.n_tests:])),
                np.sqrt(mean_squared_error(actual.iloc[-config.n_tests:], predict.iloc[-config.n_tests:])),
            ],
            "Directional Accuracy": [
                directional_accuracy_general(actual, base, config.n_forecasts),
                directional_accuracy_general(actual, predict, config.n_forecasts),
            ],
        }
    )

    forecast_df = adapter.forecast(
        data,
        config=config,
        scaler=scaler,
        criterion=criterion,
    )
    forecast_df = _normalize_prediction_frame(
        forecast_df,
        expected_horizon=config.n_forecasts,
        model_name=model_name,
    )
    forecast_df = forecast_df.rename(columns={"Predicted VN-INDEX": "Future Predictions"})

    return ForecastArtifacts(
        final_df=final_df.iloc[-config.n_tests:],
        metrics_df=metrics_df,
        forecast_df=forecast_df,
    )


def run_all_daily_forecasts(
    df: pd.DataFrame,
    *,
    config: DailyForecastConfig | None = None,
    adapters: Mapping[str, ForecastModelAdapter] | None = None,
    scaler_factory: Callable[[], object] | None = None,
    criterion_factory: Callable[[], object] | None = None,
    baseline_forecaster: BaselineForecaster | None = None,
) -> dict[str, ForecastArtifacts]:
    config = config or DailyForecastConfig()
    scaler_factory = scaler_factory or StandardScaler
    criterion_factory = criterion_factory or _default_criterion

    results: dict[str, ForecastArtifacts] = {}
    for model_name in config.model_names:
        results[model_name] = run_daily_forecast(
            df,
            model_name,
            config=config,
            scaler=scaler_factory(),
            criterion=criterion_factory(),
            adapters=adapters,
            baseline_forecaster=baseline_forecaster,
        )
    return results


def _validate_config(config: DailyForecastConfig) -> None:
    if config.n_tests <= 0:
        raise ValueError("n_tests must be positive")
    if config.n_forecasts <= 0:
        raise ValueError("n_forecasts must be positive")
    if config.n_lags <= 0:
        raise ValueError("n_lags must be positive")
    if config.seasonal_periods <= 0:
        raise ValueError("seasonal_periods must be positive")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")


def _prepare_vn_index_data(
    df: pd.DataFrame,
    config: DailyForecastConfig,
) -> pd.DataFrame:
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required VN-Index columns: {', '.join(missing_columns)}")

    data = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    if data["Date"].isna().any():
        raise ValueError("VN-Index data contains invalid Date values")

    data["VN_Index_Close"] = pd.to_numeric(data["VN_Index_Close"], errors="coerce")
    if data["VN_Index_Close"].isna().any():
        raise ValueError("VN-Index data contains invalid VN_Index_Close values")

    data = data.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    data = data.reset_index(drop=True)

    minimum_rows = max(
        config.n_tests + 1,
        config.n_tests + config.n_lags + config.n_forecasts,
    )
    if len(data) < minimum_rows:
        raise ValueError(
            "Insufficient VN-Index rows for daily forecast run: "
            f"need at least {minimum_rows}, got {len(data)}"
        )

    return data


def _resolve_adapter(
    model_name: str,
    adapters: Mapping[str, ForecastModelAdapter] | None,
) -> ForecastModelAdapter:
    available_adapters = dict(adapters or create_default_model_adapters())
    try:
        return available_adapters[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(available_adapters))
        raise ValueError(f"Unsupported forecast model '{model_name}'. Supported models: {supported}") from exc


def _normalize_prediction_frame(
    df: pd.DataFrame,
    *,
    expected_horizon: int,
    model_name: str,
) -> pd.DataFrame:
    missing_columns = [col for col in PREDICTION_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{model_name} adapter returned predictions without: {', '.join(missing_columns)}"
        )

    result = df.loc[:, list(PREDICTION_COLUMNS)].copy()
    if len(result) < expected_horizon:
        raise ValueError(
            f"{model_name} adapter returned {len(result)} predictions; "
            f"expected at least {expected_horizon}"
        )

    result = result.tail(expected_horizon).reset_index(drop=True)
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
    if result["Date"].isna().any():
        raise ValueError(f"{model_name} adapter returned invalid prediction dates")

    result["Predicted VN-INDEX"] = pd.to_numeric(
        result["Predicted VN-INDEX"],
        errors="coerce",
    )
    if result["Predicted VN-INDEX"].isna().any():
        raise ValueError(f"{model_name} adapter returned invalid predictions")

    return result


def _exponential_smoothing_forecast(
    series: pd.Series,
    n_forecasts: int,
    config: DailyForecastConfig,
) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with _suppress_model_warnings():
        baseline = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=config.seasonal_periods,
        ).fit()
        return baseline.forecast(n_forecasts).values


def _default_criterion():
    from .helper import CustomizedLoss

    return CustomizedLoss()


@contextmanager
def _suppress_model_warnings():
    with warnings.catch_warnings():
        try:
            from statsmodels.tools.sm_exceptions import ValueWarning as StatsmodelsValueWarning
        except ModuleNotFoundError:
            StatsmodelsValueWarning = None

        if StatsmodelsValueWarning is not None:
            warnings.simplefilter("ignore", category=StatsmodelsValueWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        yield
