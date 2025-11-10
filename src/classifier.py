# src/classifier.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopStockNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_classifier(X, y, epochs=30, lr=1e-3, device='cpu'):
    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    model = TopStockNet(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    for e in range(epochs):
        opt.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        opt.step()
    return model
