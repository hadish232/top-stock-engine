import torch
from torch.utils.data import TensorDataset, DataLoader
from src.models import VolatilityPredictor

def train_model(X, y, epochs=20, lr=1e-3):
    model = VolatilityPredictor(input_dim=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        opt.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        opt.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

    return model
