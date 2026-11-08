import numpy as np
import torch
import torch.nn as nn


# Small 3-layer MLP classifier for heart disease prediction that outputs a raw logit.
class HeartMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        elif x.dtype != torch.float32:
            x = x.to(torch.float32)
        return self.net(x)


# Create and initialize a new HeartMLP model with a fixed seed for reproducible weights.
def make_model(seed=42):
    torch.manual_seed(seed)
    return HeartMLP()


# Extract copies of all model parameters as a list of numpy arrays in a fixed order.
def get_weights(model):
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


# Load a list of numpy weight arrays back into the model parameters in place.
def set_weights(model, weights):
    with torch.no_grad():
        for p, w in zip(model.parameters(), weights):
            p.copy_(torch.from_numpy(np.asarray(w)).to(dtype=p.dtype, device=p.device))
