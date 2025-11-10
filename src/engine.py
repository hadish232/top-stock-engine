# src/engine.py
import os, numpy as np, pandas as pd
import torch
from typing import List
from src.data_loader import fetch_history_yf, load_sector_plan_files
from src.features import compute_basic_features
from src.hawkes_evt import batch_hawkes, adaptive_evt_tail_prob
from src.gnn import SectorGNN, PYGEOM_OK
from src.classifier import TopStockNet
from sklearn.preprocessing import StandardScaler

def build_universe(tickers: List[str], period="1y", interval="1d"):
    hist = {t: fetch_history_yf(t, period=period, interval=interval) for t in tickers}
    return hist

def prepare_matrix(hist):
    # Build aligned close and vol matrices (days x n_stocks)
    # pick intersection of dates
    dfs = list(hist.values())
    common_idx = set(dfs[0]['Date'])
    for df in dfs[1:]:
        common_idx = common_idx.intersection(set(df['Date']))
    common_idx = sorted(list(common_idx))
    n_days = len(common_idx); n_stocks = len(dfs)
    close = np.zeros((n_days, n_stocks), dtype=float)
    vol = np.zeros_like(close)
    for j,t in enumerate(hist.keys()):
        df = hist[t].set_index('Date').reindex(common_idx)
        close[:, j] = df['Close'].values
        vol[:, j] = df['Volume'].values
    return close, vol, common_idx, list(hist.keys())

def engine_run(tickers: List[str], sector_csv="nse_sector.csv", plan_csv="company_plan_scores.csv"):
    # 1) Load history
    hist = build_universe(tickers, period="2y", interval="1d")
    close, vol, dates, stocks = prepare_matrix(hist)
    # 2) Compute features per stock (last N days)
    n_days, n_stocks = close.shape
    features = []
    for j,stk in enumerate(stocks):
        df = pd.DataFrame({
            'Date': dates,
            'Open': close[:,j], 'High': close[:,j], 'Low': close[:,j],
            'Close': close[:,j], 'Volume': vol[:,j]
        })
        f = compute_basic_features(df)
        # take last row features
        lr = f.iloc[-1]
        features.append([lr['ret'], lr['ewma_vol'], lr['mom5'], lr['vol_z']])
    X = np.vstack(features)  # n_stocks x features
    # 3) compute hawkes & EVT tail probs
    hawkes = batch_hawkes(vol, alpha=0.8, beta=0.6)
    tail = adaptive_evt_tail_prob((np.diff(close, axis=0) / (close[:-1] + 1e-9)), threshold=0.03, window=252)
    # 4) combine features + hawkes + tail
    X_full = np.hstack([X, hawkes.reshape(-1,1), tail.reshape(-1,1)])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_full)
    # 5) simple heuristic fusion score + optional classifier (here we use logic)
    # fusion: weighted linear combination
    w = np.array([0.15, 0.25, 0.2, 0.15, 0.15, 0.10])  # sum 1
    score = 1/(1+np.exp(- (Xs @ w)))  # sigmoid mapping to [0,1]
    # 6) build DataFrame and return top stocks with probability
    out = pd.DataFrame({
        'Stock': stocks,
        'Score': score,
        'Hawkes': hawkes,
        'TailProb': tail
    })
    out = out.sort_values('Score', ascending=False).reset_index(drop=True)
    return out
