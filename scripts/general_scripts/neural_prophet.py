from neuralprophet import NeuralProphet, set_random_seed, set_log_level
import pandas as pd

set_random_seed(0)
# Disable logging messages unless there is an error
set_log_level("ERROR")

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