from neuralprophet import NeuralProphet, set_random_seed, set_log_level
from neuralprophet import configure as neuralprophet_configure
import pandas as pd
import torch
from contextlib import contextmanager
import inspect

set_random_seed(0)
# Disable logging messages unless there is an error
set_log_level("ERROR")


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
