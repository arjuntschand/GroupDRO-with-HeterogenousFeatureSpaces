from .cnn28 import CNN28
from .cnn32 import CNN32
from .resnet_visual import ResNetVisualEncoder, SimpleCNNVisualEncoder
from .text_encoder import CharCNNTextEncoder, TransformerTextEncoder, MLPTextEncoder

ENCODER_REGISTRY = {
    # Original MNIST/USPS encoders
    "cnn28": CNN28,
    "cnn32": CNN32,
    
    # TextCaps visual encoders
    "resnet_visual": ResNetVisualEncoder,
    "simple_cnn_visual": SimpleCNNVisualEncoder,
    
    # TextCaps text encoders
    "char_cnn_text": CharCNNTextEncoder,
    "transformer_text": TransformerTextEncoder,
    "mlp_text": MLPTextEncoder,
}
