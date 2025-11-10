# src/hawkes_evt.py
import numpy as np
import torch
from scipy.stats import genpareto

def hawkes_intensity_from_volume(vol_series: np.ndarray, alpha=0.8, beta=0.6):
    """
    vol_series: 1D numpy array of past volumes (most recent last).
    Returns a scalar intensity. Vectorize across stocks at call site.
    Simple exponential-decay kernel proxy.
    """
    n = len(vol_series)
    if n == 0:
        return 0.0
    dt = np.arange(n, 0, -1)
    weights = alpha * np.exp(-beta * dt)
    return float(np.sum(weights * vol_series))

def batch_hawkes(vol_matrix: np.ndarray, alpha=0.8, beta=0.6):
    # vol_matrix: days x n_stocks
    n_stocks = vol_matrix.shape[1]
    out = np.zeros(n_stocks, dtype=float)
    for i in range(n_stocks):
        out[i] = hawkes_intensity_from_volume(vol_matrix[:, i], alpha=alpha, beta=beta)
    return out

def adaptive_evt_tail_prob(returns_matrix: np.ndarray, threshold=0.03, window=252):
    """
    returns_matrix: days x n_stocks
    Fit GPD on standardized excesses in rolling window for each stock.
    Returns vector of tail survival probabilities P(|ret| >= threshold)
    """
    days, n = returns_matrix.shape
    probs = np.zeros(n, dtype=float)
    if days < 20:
        return probs + 0.01
    data = returns_matrix[-window:, :]
    for i in range(n):
        tails = np.abs(data[:, i])
        excess = np.clip(tails - threshold, 0, None)
        if np.sum(excess) > 0:
            try:
                c, loc, scale = genpareto.fit(excess, floc=0)
                probs[i] = genpareto.sf(threshold, c, loc=0, scale=scale)
            except Exception:
                probs[i] = 0.01
        else:
            probs[i] = 0.01
    return probs
