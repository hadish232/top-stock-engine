import torch
import torch.nn as nn

class VolatilityPredictor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)
