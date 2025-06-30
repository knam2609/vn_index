import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from statsmodels.tsa.arima.model import ARIMA

def fourier_decompose(y):
    # Step 1: Build deterministic features
    fourier_yearly = CalendarFourier(freq='A', order=3)
    fourier_weekly = CalendarFourier(freq='W', order=1)

    # Before creating dp
    if y.index.freq is None:
        y = y.asfreq('B')  # or use date_range() fallback
        y = y.ffill().bfill()  # fill forward, then backward
        print(y)
    
    dp = DeterministicProcess(
        index=y.index,
        constant=True,
        order=3,  # linear + quadratic trend
        seasonal=False,
        additional_terms=[fourier_yearly, fourier_weekly],
        drop=True
    )
    
    X = dp.in_sample()
    model = LinearRegression()
    model.fit(X, y)
    y_pred = pd.Series(model.predict(X), index=y.index)

    # Step 2: Manually compute trend using trend-related columns only
    coefs = model.coef_
    colnames = X.columns
    intercept = model.intercept_

    trend_terms = [name for name in colnames if "trend" in name or name == "const"]
    trend_X = X[trend_terms]
    trend_coefs = [coefs[colnames.get_loc(name)] for name in trend_terms]

    # Compute trend using only the matching coefficients
    trend = pd.Series(np.dot(trend_X.values, trend_coefs) + intercept, index=y.index)

    # Step 3: Derive seasonal and residual
    seasonal = y_pred - trend
    residuals = y - y_pred

    fig, axs = plt.subplots(4, 1, figsize=(15, 10), sharex=True)

    axs[0].plot(y, label='Observed')
    axs[0].set_title('Original Series')

    axs[1].plot(trend, label='Trend', color='orange')
    axs[1].set_title('Trend')

    axs[2].plot(seasonal, label='Seasonality', color='green')
    axs[2].set_title('Combined Weekly + Yearly Seasonality')

    axs[3].plot(residuals, label='Residuals', color='red')
    axs[3].set_title('Residuals')

    plt.tight_layout()
    plt.show()

    return trend, seasonal, residuals, model, dp, trend_terms, trend_coefs, intercept

def forecast_from_fourier_decompose(y, n_steps=2, arima_order=(5, 0, 0), plot=False):
    """
    Forecasts trend + seasonal + residuals using an existing fourier_decompose().
    """
    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError("Input series must have a DatetimeIndex for Fourier decomposition.")
    trend, seasonal, residuals, model, dp, trend_terms, trend_coefs, intercept = fourier_decompose(y)

    # Step 1: Forecast future trend + seasonality
    X_future = dp.out_of_sample(steps=n_steps)
    future_index = X_future.index
    trend_X_future = X_future[trend_terms]
    trend_forecast = pd.Series(np.dot(trend_X_future.values, trend_coefs) + intercept, index=future_index)
    seasonal_forecast = pd.Series(model.predict(X_future), index=future_index) - trend_forecast

    # Step 2: Forecast residuals using ARIMA
    resid_model = ARIMA(residuals, order=arima_order).fit()
    resid_forecast = pd.Series(resid_model.forecast(steps=n_steps), index=future_index)

    # Step 3: Combine components
    final_forecast = trend_forecast + seasonal_forecast + resid_forecast

    if plot:
        # Optional: plot
        plt.figure(figsize=(14, 6))
        plt.plot(y, label="Historical")
        plt.plot(final_forecast, label="Forecast", color='red')
        plt.title("VN-Index Forecast: Trend + Seasonality + Residuals")
        plt.grid(True)
        plt.legend()
        plt.show()

    return trend, seasonal, residuals, final_forecast
