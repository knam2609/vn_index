from neuralprophet import NeuralProphet, set_random_seed, set_log_level
from neuralprophet import configure as neuralprophet_configure
import numpy as np
import pandas as pd
import torch
from contextlib import contextmanager
import inspect

set_random_seed(0)
# Disable logging messages unless there is an error
set_log_level("ERROR")


def _patch_neuralprophet_pandas3_compat():
    """
    NeuralProphet 0.9.0 still relies on pandas behaviors that changed in 3.x.
    Patch only the affected upstream helpers with equivalent implementations.
    """
    try:
        from neuralprophet import df_utils as neuralprophet_df_utils
    except Exception:
        return

    try:
        from neuralprophet.data import process as neuralprophet_process
    except Exception:
        neuralprophet_process = None

    try:
        import neuralprophet.forecaster as neuralprophet_forecaster
    except Exception:
        neuralprophet_forecaster = None

    current_get_freq_dist = getattr(neuralprophet_df_utils, "get_freq_dist", None)
    if current_get_freq_dist is None:
        return
    if getattr(current_get_freq_dist, "_vn_index_pandas3_compat", False):
        get_freq_dist_patched = True
    else:
        get_freq_dist_patched = False

    if not get_freq_dist_patched:
        def _get_freq_dist_compat(ds_col):
            converted_ds = pd.Series(
                pd.to_datetime(ds_col, utc=True).to_numpy(dtype="datetime64[ns]").astype(np.int64),
                index=getattr(ds_col, "index", None),
            )
            diff_ds = np.unique(converted_ds.diff(), return_counts=True)
            return diff_ds

        _get_freq_dist_compat._vn_index_pandas3_compat = True
        neuralprophet_df_utils.get_freq_dist = _get_freq_dist_compat

    if neuralprophet_process is None:
        return

    current_handle_missing_data = getattr(neuralprophet_process, "_handle_missing_data", None)
    if current_handle_missing_data is None:
        return
    if getattr(current_handle_missing_data, "_vn_index_pandas3_compat", False):
        return

    def _handle_missing_data_compat(
        df,
        freq,
        n_lags,
        n_forecasts,
        config_missing,
        config_regressors=None,
        config_lagged_regressors=None,
        config_events=None,
        config_seasonality=None,
        predicting=False,
    ):
        df, _, _, _ = neuralprophet_process.df_utils.prep_or_copy_df(df)

        if n_lags == 0 and not predicting:
            df_na_dropped = df.dropna(subset=["y"])
            n_dropped = len(df) - len(df_na_dropped)
            if n_dropped > 0:
                df = df_na_dropped
                neuralprophet_process.log.info(f"Dropped {n_dropped} rows with NaNs in 'y' column.")

        if n_lags > 0:
            df_grouped = (
                df.groupby("ID")
                .apply(lambda x: x.set_index("ds").resample(freq).asfreq())
                .drop(columns=["ID"], errors="ignore")
            )
            n_missing_dates = len(df_grouped) - len(df)
            if n_missing_dates > 0:
                df = df_grouped.reset_index()
                neuralprophet_process.log.info(f"Added {n_missing_dates} missing dates.")

        if config_regressors is not None and config_regressors.regressors is not None:
            last_valid_index = df.groupby("ID")[list(config_regressors.regressors.keys())].apply(
                lambda x: x.last_valid_index()
            )
            df_dropped = df.groupby("ID", group_keys=False).apply(lambda x: x.loc[: last_valid_index[x.name]])
            n_dropped = len(df) - len(df_dropped)
            if n_dropped > 0:
                df = df_dropped
                neuralprophet_process.log.info(f"Dropped {n_dropped} rows at the end with NaNs in future regressors.")

        dropped_trailing_y = False
        if df["y"].isna().any():
            last_valid_index = df.groupby("ID")["y"].apply(lambda x: x.last_valid_index())
            df_dropped = df.groupby("ID", group_keys=False).apply(lambda x: x.loc[: last_valid_index[x.name]])
            n_dropped = len(df) - len(df_dropped)
            if n_dropped > 0:
                dropped_trailing_y = True
                df_to_add = df.groupby("ID", group_keys=False).apply(lambda x: x.loc[last_valid_index[x.name] + 1 :])
                df = df_dropped
                neuralprophet_process.log.info(f"Dropped {n_dropped} rows at the end with NaNs in 'y' column.")

        if config_missing.impute_missing:
            data_columns = []
            if n_lags > 0:
                data_columns.append("y")
            if config_lagged_regressors is not None:
                data_columns.extend(config_lagged_regressors.keys())
            if config_regressors is not None and config_regressors.regressors is not None:
                data_columns.extend(config_regressors.regressors.keys())
            if config_events is not None:
                data_columns.extend(config_events.keys())
            conditional_cols = []
            if config_seasonality is not None:
                conditional_cols = list(
                    set(
                        [
                            value.condition_name
                            for key, value in config_seasonality.periods.items()
                            if value.condition_name is not None
                        ]
                    )
                )
                data_columns.extend(conditional_cols)
            for column in data_columns:
                sum_na = df[column].isna().sum()
                if sum_na > 0:
                    neuralprophet_process.log.warning(f"{sum_na} missing values in column {column} were detected in total. ")
                    if config_events is not None and column in config_events.keys():
                        df[column].fillna(0, inplace=True)
                        remaining_na = 0
                    else:
                        df.loc[:, column], remaining_na = neuralprophet_process.df_utils.fill_linear_then_rolling_avg(
                            df[column],
                            limit_linear=config_missing.impute_linear,
                            rolling=config_missing.impute_rolling,
                        )
                    neuralprophet_process.log.info(f"{sum_na - remaining_na} NaN values in column {column} were auto-imputed.")
                    if remaining_na > 0:
                        neuralprophet_process.log.warning(
                            f"More than {2 * config_missing.impute_linear + config_missing.impute_rolling} consecutive "
                            f"missing values encountered in column {column}. {remaining_na} NA remain after auto-imputation. "
                        )
        if dropped_trailing_y and predicting:
            df = pd.concat([df, df_to_add])
            if config_seasonality is not None and len(conditional_cols) > 0:
                df[conditional_cols] = df[conditional_cols].ffill()  # type: ignore
        return df

    _handle_missing_data_compat._vn_index_pandas3_compat = True
    neuralprophet_process._handle_missing_data = _handle_missing_data_compat
    if neuralprophet_forecaster is not None:
        neuralprophet_forecaster._handle_missing_data = _handle_missing_data_compat


@contextmanager
def torch_load_compat_context():
    """
    NeuralProphet/Lightning can restore temporary checkpoints during fit().
    PyTorch 2.6 changed torch.load default weights_only=True, which breaks
    those restores for older checkpoint formats. For trusted local checkpoints,
    force weights_only=False during NeuralProphet training.
    """
    original_torch_load = torch.load
    original_serialization_load = torch.serialization.load
    cloud_io_module = None
    original_cloud_io_load = None
    original_cloud_io_pl_load = None
    pl_cloud_io_module = None
    original_pl_cloud_io_load = None

    def _torch_load_with_compat(*args, **kwargs):
        # Always force full checkpoint load for trusted local training artifacts.
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    if hasattr(torch.serialization, "add_safe_globals"):
        safe_types = [
            obj
            for _, obj in inspect.getmembers(neuralprophet_configure)
            if isinstance(obj, type) and obj.__module__.startswith("neuralprophet")
        ]
        if safe_types:
            torch.serialization.add_safe_globals(safe_types)

    torch.load = _torch_load_with_compat
    torch.serialization.load = _torch_load_with_compat

    # Lightning may keep internal loader aliases; patch them if present.
    try:
        import lightning_fabric.utilities.cloud_io as cloud_io
        cloud_io_module = cloud_io
        if hasattr(cloud_io_module, "_load"):
            original_cloud_io_load = cloud_io_module._load
            cloud_io_module._load = _torch_load_with_compat
        if hasattr(cloud_io_module, "pl_load"):
            original_cloud_io_pl_load = cloud_io_module.pl_load
            cloud_io_module.pl_load = _torch_load_with_compat
    except Exception:
        pass

    try:
        import pytorch_lightning.utilities.cloud_io as pl_cloud_io
        pl_cloud_io_module = pl_cloud_io
        if hasattr(pl_cloud_io_module, "load"):
            original_pl_cloud_io_load = pl_cloud_io_module.load
            pl_cloud_io_module.load = _torch_load_with_compat
    except Exception:
        pass

    try:
        yield
    finally:
        torch.load = original_torch_load
        torch.serialization.load = original_serialization_load
        if cloud_io_module is not None:
            if original_cloud_io_load is not None:
                cloud_io_module._load = original_cloud_io_load
            if original_cloud_io_pl_load is not None:
                cloud_io_module.pl_load = original_cloud_io_pl_load
        if pl_cloud_io_module is not None and original_pl_cloud_io_load is not None:
            pl_cloud_io_module.load = original_pl_cloud_io_load

def modify(df, forecast=5):
    # figure out where 'yhat1' lives
    col_idx = df.columns.get_loc('yhat1')

    # loop 1…9
    for i in range(1, forecast):
        # compute which yhatN we need
        which = forecast - i + 1          # 10→1, 9→2, …, 2→9
        val   = df.iloc[-i][f'yhat{which}'] 
        # row number = -i (so i=1 → last row, i=9 → 9th-from-last)
        df.iat[-i, col_idx] = val

    # j = 0
    # for i in range(forecast+1, forecast+6):
    #     which = min(5, forecast - j +1)
    #     print(which)
    #     print(df.iloc[-i]['ds'])

    #     val = df.iloc[-i][f'yhat{which}']
    #     df.iat[-i, col_idx] = val
    #     j += 1

def neural_prophet_model(df, n_lags, n_forecasts):
    _patch_neuralprophet_pandas3_compat()

    # 1) Instantiate a NeuralProphet model that learns from the past 60 days (n_lags)
    #    to forecast the next 10 days (n_forecasts), with trend + all seasonalities on.
    m = NeuralProphet(
        n_lags=n_lags,                 # look back this many days as inputs
        n_forecasts=n_forecasts,            # predict this many days ahead
        yearly_seasonality=True,   # model an annual cycle
        weekly_seasonality=True,   # model day-of-week cycle
        daily_seasonality=False,    # if intra-day data; set False for daily only
        seasonality_mode='additive',
        n_changepoints=10,
        # you can also tweak trend flexibility:
        trend_reg=0.5,                  
        trend_reg_threshold=True,
        normalize='standardize',
        drop_missing=True,
    )
    m.set_plotting_backend("plotly-static")

    df = (df[['Date', 'VN_Index_Close']].rename(columns={
          'Date': 'ds',
          'VN_Index_Close': 'y'
      }))

    # convert the 'ds' column to datetime
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.dropna(subset=['ds', 'y']).sort_values('ds')
    df = df.drop_duplicates(subset=['ds'], keep='last').reset_index(drop=True)

    # 2) compute how many rows you need for validation
    n_val = m.n_lags + m.n_forecasts  # must equal 60 + 10 = 70

    # 3) split your DataFrame by row‐count
    train_df = df.iloc[:-n_val]
    val_df   = df.iloc[-n_val:]

    # 4) fit using the fixed‐size validation set
    with torch_load_compat_context():
        metrics = m.fit(
            train_df,
            freq="B",
            validation_df=val_df,
            progress="plot",
        )

    future = m.make_future_dataframe(df, periods=m.n_forecasts, n_historic_predictions=False)
    forecast = m.predict(future, auto_extend=False, raw=False)

    # print(forecast)
    modify(forecast, m.n_forecasts)
    # Visualize the forecast
    m.highlight_nth_step_ahead_of_each_forecast(step_number=1)
    # m.plot(forecast)

    result = forecast[['ds', 'yhat1']].tail(m.n_forecasts).rename(columns={
          'ds': 'Date',
          'yhat1': 'Predicted VN-INDEX'
      })
    
    print(result)
    return result
