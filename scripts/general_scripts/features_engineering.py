# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.graphics.tsaplots import plot_acf
from .fourier_arima import forecast_from_fourier_decompose

def compute_RSI(series, window=14):
    """
    Compute the Relative Strength Index (RSI) for a time-series.
    
    Args:
        series (pd.Series): Series of prices.
        window (int): Window size.
    
    Returns:
        pd.Series: RSI values.
    """
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))


def lag_features_indicators(df, numerical_columns):
    """
    Generate lag features, moving averages, RSI, MACD, volatility, seasonality,
    and interaction features.
    
    Args:
        df (pd.DataFrame): Input data.
        numerical_columns (list): List of numerical column names.
    
    Returns:
        pd.DataFrame: DataFrame with additional features.
    """

    copy_df = df.copy()
    # Lagged Features
    lag_days = [1, 2, 3, 5, 10]
    for col in numerical_columns:
        for lag in lag_days:
            copy_df[f'{col}_Lag{lag}'] = copy_df[col].shift(lag)
    
    # Moving Averages and Exponential Moving Averages
    for col in numerical_columns:
        copy_df[f'{col}_SMA_10'] = copy_df[col].rolling(window=10).mean()
        copy_df[f'{col}_SMA_20'] = copy_df[col].rolling(window=20).mean()
        copy_df[f'{col}_EMA_10'] = copy_df[col].ewm(span=10, adjust=False).mean()
        copy_df[f'{col}_EMA_20'] = copy_df[col].ewm(span=20, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    for col in numerical_columns:
        copy_df[f'{col}_RSI_14'] = compute_RSI(copy_df[col])
    
    # Moving Average Convergence Divergence (MACD)
    for col in numerical_columns:
        copy_df[f'{col}_EMA_12'] = copy_df[col].ewm(span=12, adjust=False).mean()
        copy_df[f'{col}_EMA_26'] = copy_df[col].ewm(span=26, adjust=False).mean()
        copy_df[f'{col}_MACD'] = copy_df[f'{col}_EMA_12'] - copy_df[f'{col}_EMA_26']
    
    # Additional Feature: Rolling Standard Deviation for Volatility
    for col in numerical_columns:
        copy_df[f'{col}_RollingStd_10'] = copy_df[col].rolling(window=10).std()
    
    # Interaction Feature: Ratio of EMA_10 to EMA_20
    for col in numerical_columns:
        copy_df[f'{col}_EMA_Ratio'] = copy_df[f'{col}_EMA_10'] / copy_df[f'{col}_EMA_20']
    
    # Drop NA values caused by shifting and rolling
    copy_df.dropna(inplace=True)
    
    return copy_df

def quicky_data(df, period, decomposition_tech='stl', plot=False):
    """
    Preprocess the data by converting the 'Date' column to datetime,
    setting it as index, and dropping unnecessary columns.
    
    Args:
        df (pd.DataFrame): Raw data.
        period (int): Seasonality.
    
    Returns:
        pd.DataFrame: Preprocessed data.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    if 'Index' in df.columns:
        df.drop(columns=['Index'], inplace=True)

    # ---- decomposition ----
    series = df.iloc[:,0]
    if period < 2:
        # Apply HP filter to extract trend (cycle is the residual)
        cycle, trend = hpfilter(series, lamb=129600)  # lamb=1600 is standard for monthly data

        df["trend"] = trend
        df["seasonal"] = 0  # explicitly ignore seasonality
        df["residual"] = cycle  # residuals = cyclical part
        if plot:
            # 📊 Plot
            plt.figure(figsize=(12,6))
            plt.plot(series, label='Original', color='black', linewidth=1)
            plt.plot(trend, label='Trend (HP)', color='blue', linestyle='--')
            plt.plot(cycle, label='Cycle (Residual)', color='red', linestyle=':')
            plt.title("Hodrick-Prescott Filter Decomposition")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.show()
    else:
        if decomposition_tech == 'stl':
            stl = STL(series, period=period, robust=True)
            res = stl.fit()
            if plot:
                res.plot()
            df["trend"] = res.trend
            df["seasonal"] = res.seasonal
            df["residual"] = res.resid
        
        else:
            y = df[df.columns[0]]
            print(y)
            df["trend"], df["seasonal"], df["residual"], forecast = forecast_from_fourier_decompose(y, plot=plot)
            print(forecast)

        seasonality_strength = 1 - (df["residual"].var() / (df["residual"].var() + df["seasonal"].var()))
        print(f"Seasonality Strength: {seasonality_strength:.2f}")

    if plot:
        plt.figure(figsize=(12, 4))
        plot_acf(df["residual"], lags=60, zero=False)
        plt.title("ACF of Residuals")
        plt.show()

    # ---- date embeddings ----
    # day of week: 0 = Monday, … 6 = Sunday
    dow = df.index.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    
    # month: 1 = Jan, … 12 = Dec
    mth = df.index.month - 1  # shift to 0–11
    df['month_sin'] = np.sin(2 * np.pi * mth / 12)
    df['month_cos'] = np.cos(2 * np.pi * mth / 12)

    return df

def select_top_correlated_features(df, target_col="VN_Index_Close", top_k=10):
    """
    Selects top_k most correlated features with the target,
    excluding decomposition and time-based features.

    Args:
        df (pd.DataFrame): Full dataset.
        target_col (str): Name of target column.
        top_k (int): Number of top features to select.

    Returns:
        pd.DataFrame: Subset with target_col + selected features.
    """
    # Features to exclude based on name
    exclude_keywords = {"trend", "seasonal", "residual", "dow_sin", "dow_cos", "month_sin", "month_cos"}

    def is_valid_feature(col):
        return (
            col != target_col and
            not any(keyword in col.lower() for keyword in exclude_keywords)
        )

    # Filter valid features
    candidate_features = [col for col in df.columns if is_valid_feature(col)]

    # Compute correlations
    corr_series = df[candidate_features].corrwith(df[target_col]).abs()

    # Select top-k
    top_features = corr_series.sort_values(ascending=False).head(top_k).index.tolist()

    print(f"Top {top_k} correlated features (excluding decomposition/date):")
    print(top_features)

    return df[[target_col] + top_features]
