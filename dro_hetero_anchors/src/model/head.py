import torch.nn as nn
import torch

class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, z: torch.Tensor):
        return self.fc(z)

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
    def forward(self, z: torch.Tensor):
        return self.fc2(torch.relu(self.fc1(z)))
