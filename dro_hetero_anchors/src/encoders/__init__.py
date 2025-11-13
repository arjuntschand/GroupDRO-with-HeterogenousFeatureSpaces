from .cnn28 import CNN28
from .cnn32 import CNN32

ENCODER_REGISTRY = {
    "cnn28": CNN28,
    "cnn32": CNN32,
}
