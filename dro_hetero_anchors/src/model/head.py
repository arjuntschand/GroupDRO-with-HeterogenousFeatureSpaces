import torch.nn as nn
import torch

class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, z: torch.Tensor):
        return self.fc(z)

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_classes)
    def forward(self, z: torch.Tensor):
        z = torch.relu(self.fc1(z))
        z = self.dropout(z)
        return self.fc2(z)


class PerGroupHead(nn.Module):
    """One head per group. With per-group encoders and ERM this makes every group an independent
    model (no parameter is shared), trained inside the same loop so the optimiser, schedule,
    batching and epoch selection match the other arms. Comparator for the no-overlap experiments."""
    def __init__(self, make_head, num_groups: int):
        super().__init__()
        self.heads = nn.ModuleList([make_head() for _ in range(num_groups)])
    def forward(self, z: torch.Tensor, g: torch.Tensor):
        out = None
        for gid, h in enumerate(self.heads):
            m = (g == gid)
            if m.any():
                o = h(z[m])
                if out is None:
                    out = z.new_zeros(z.size(0), o.size(1))
                out[m] = o
        return out


def apply_head(head, z, g):
    return head(z, g) if isinstance(head, PerGroupHead) else head(z)
