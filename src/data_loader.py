# src/data_loader.py
import os, time, json
import pandas as pd
import yfinance as yf
from typing import List, Tuple

CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_history_yf(ticker: str, period="1y", interval="1d", force=False) -> pd.DataFrame:
    """Download and cache historical data for ticker (ticker without .NS)"""
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_{period}_{interval}.parquet")
    if os.path.exists(cache_path) and not force:
        return pd.read_parquet(cache_path)
    df = yf.download(f"{ticker}.NS", period=period, interval=interval, auto_adjust=True, progress=False)
    df = df.reset_index().rename(columns={'Date':'Date'})
    df.to_parquet(cache_path)
    time.sleep(0.25)
    return df

def fetch_universe_hist(tickers: List[str], period="1y", interval="1d", force=False):
    out = {}
    for t in tickers:
        try:
            out[t] = fetch_history_yf(t, period=period, interval=interval, force=force)
        except Exception as e:
            print("err", t, e)
    return out

def load_sector_plan_files(sector_csv="nse_sector.csv", plan_csv="company_plan_scores.csv") -> Tuple[pd.DataFrame,pd.DataFrame]:
    sector = pd.read_csv(sector_csv)
    plan = pd.read_csv(plan_csv)
    return sector, plan

