import torch.nn as nn
import torch.nn.functional as F

class CNN28(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.adapt = nn.AdaptiveAvgPool2d((7, 7))  # handle any H×W
        self.fc = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool(x)
        x = F.relu(self.conv2(x)); x = self.pool(x)
        x = self.adapt(x)          # ensure 7×7 before FC
        x = x.flatten(1)
        return self.fc(x)
