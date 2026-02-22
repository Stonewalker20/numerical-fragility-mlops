import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self, in_dim=28*28, hidden=256, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        # x: (B, 1, 28, 28)
        x = x.view(x.size(0), -1)
        return self.net(x)

