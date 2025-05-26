# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================
import pandas as pd
from statsmodels.tsa.seasonal import STL

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

    # Identify and exclude date‑embedding columns
    date_embeddings = {
        'dow_sin', 'dow_cos',
        'month_sin', 'month_cos'
    }
    numerical_columns = [col for col in numerical_columns if col not in date_embeddings]

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


def quicky_data(df):
    """
    Preprocess the data by converting the 'Date' column to datetime,
    setting it as index, and dropping unnecessary columns.
    
    Args:
        df (pd.DataFrame): Raw data.
    
    Returns:
        pd.DataFrame: Preprocessed data.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    if 'Index' in df.columns:
        df.drop(columns=['Index'], inplace=True)

    # ---- decomposition ----
    series = df.iloc[:,0]
    stl = STL(series, period=261, robust=True)
    res = stl.fit()
    res.plot()

    df['trend'] = res.trend
    df['seasonal'] = res.seasonal
    df['residual'] = res.resid

    return df

def select_features_by_correlation(df, target_col="VN_Index_Close", train_ratio=0.9, corr_threshold=0.05):
    """
    Splits the DataFrame by time (first train_ratio% of rows is 'training'),
    calculates correlation of each feature with the target on TRAIN rows only,
    and returns the subset of columns (target + selected features).
    
    Args:
        df (pd.DataFrame): Full dataset (includes the target column).
        target_col (str): Target column name, default = "VN_Index_Close".
        train_ratio (float): Proportion of data used for 'training'.
        corr_threshold (float): Minimum absolute correlation needed to keep a feature.
    
    Returns:
        pd.DataFrame: A filtered DataFrame with only 'target_col' + selected features.
    """
    # Sort by index if needed (assuming your index is Date or similar)
    df = df.sort_index()
    n_train = int(len(df) * train_ratio)
    
    # TRAIN portion (first 90% by default)
    df_train = df.iloc[:n_train]
    
    # Identify all potential features (exclude the target itself)
    all_features = [col for col in df.columns if col != target_col]
    
    # Calculate absolute correlation with the target on the training portion only
    corr_series = df_train[all_features].corrwith(df_train[target_col]).abs()
    
    # Filter by threshold
    selected_features = corr_series[corr_series >= corr_threshold].index.tolist()
    
    print(f"Features with abs(corr) >= {corr_threshold}:")
    print(selected_features)
    
    # Return only target + selected features
    return df[[target_col] + selected_features]