# src/training/data_processing.py

import pandas as pd

def create_features(raw_price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame of raw daily close prices for multiple tickers and
    engineers a full feature set for the ETF trend prediction model.
    """
    df = raw_price_df.copy()
    
    # --- 1. Define Target Variable (for training) ---
    # Only create the target if it's not already there (i.e., during training)
    if 'target' not in df.columns:
        df['target'] = (df['SPY'].shift(-1) > df['SPY']).astype(int)

    # --- 2. Create Return-Based Features ---
    all_tickers = [col for col in df.columns if col != 'target']
    returns_df = df[all_tickers].pct_change()

    # --- 3. Engineer Technical Indicators & Lagged Features ---
    df['spy_ma_50'] = df['SPY'].rolling(window=50).mean()
    df['spy_ma_200'] = df['SPY'].rolling(window=200).mean()
    
    delta = df['SPY'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['spy_rsi'] = 100 - (100 / (1 + rs))

    for ticker in all_tickers:
        for lag in [1, 3, 5, 10]:
            df[f'{ticker}_return_lag_{lag}'] = returns_df[ticker].shift(lag)

    df['spy_vol_30'] = returns_df['SPY'].rolling(window=30).std()
    
    # --- 4. Assemble the Final Dataset ---
    feature_columns = [
        'spy_ma_50', 'spy_ma_200', 'spy_rsi', 'spy_vol_30'
    ] + [col for col in df.columns if '_return_lag_' in col]
    
    # For live prediction, we don't have a target
    if 'target' in df.columns:
        final_dataset = df[feature_columns + ['target']]
    else:
        final_dataset = df[feature_columns]

    final_dataset.dropna(inplace=True)

    return final_dataset