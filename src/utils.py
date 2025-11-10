import torch
import numpy as np

def to_tensor(df, features, target):
    X = torch.tensor(df[features].values, dtype=torch.float32)
    y = torch.tensor(df[target].values, dtype=torch.float32).view(-1, 1)
    return X, y
