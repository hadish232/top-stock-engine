# src/features.py
import numpy as np
import pandas as pd

def compute_basic_features(df: pd.DataFrame, ewma_span=20):
    """
    Input: df with Date, Open, High, Low, Close, Volume
    Returns: df with returns, log-returns, realized vol (rolling), EWMA vol, momentum, vwap proxy
    """
    df = df.copy().sort_values("Date")
    df['ret'] = df['Close'].pct_change()
    df['logret'] = np.log1p(df['ret'])
    # realized volatility (rolling std of returns)
    df['rv20'] = df['ret'].rolling(20).std()
    # EWMA volatility
    df['ewma_var'] = df['ret'].ewm(span=ewma_span).var()
    df['ewma_vol'] = np.sqrt(df['ewma_var'])
    # momentum
    df['mom5'] = df['ret'].rolling(5).sum()
    # volume zscore
    df['vol_z'] = (df['Volume'] - df['Volume'].rolling(60).mean()) / (df['Volume'].rolling(60).std() + 1e-9)
    # vwap proxy: (High+Low+Close)/3 as typical price
    df['typ_price'] = (df['High'] + df['Low'] + df['Close']) / 3.0
    return df.dropna().reset_index(drop=True)
