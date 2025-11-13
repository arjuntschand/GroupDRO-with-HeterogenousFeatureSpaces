import random
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from rich.console import Console

console = Console()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

@dataclass
class Meter:
    total: float = 0.0
    n: int = 0

    def update(self, val: float, k: int = 1):
        self.total += float(val) * k
        self.n += k

    @property
    def avg(self) -> float:
        return self.total / max(1, self.n)

def params_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
