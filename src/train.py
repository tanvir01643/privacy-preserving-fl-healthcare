import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Train a PyTorch model in place using mini-batch Adam optimization and BCEWithLogitsLoss.
def train_model(model, X, y, epochs=100, lr=0.01, batch_size=32, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)

    X_tensor = torch.tensor(X_arr, dtype=torch.float32)
    y_tensor = torch.tensor(y_arr, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []
    model.train()

    for _ in range(epochs):
        batch_losses = []
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        avg_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        epoch_losses.append(avg_loss)

    return epoch_losses


# Predict positive-class probabilities using sigmoid on model output logits without tracking gradients.
def predict_scores(model, X):
    model.eval()
    X_arr = np.asarray(X, dtype=np.float32)
    X_tensor = torch.tensor(X_arr, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    return probs
