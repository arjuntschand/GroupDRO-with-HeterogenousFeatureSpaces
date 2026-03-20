from .cnn28 import CNN28
from .cnn32 import CNN32
from .resnet_visual import ResNetVisualEncoder, SimpleCNNVisualEncoder
from .text_encoder import CharCNNTextEncoder, TransformerTextEncoder, MLPTextEncoder
from .tabular_encoder import MLPTabularEncoder, MLPTabularEncoderLarge, MLPTabularEncoderLN, MLPTabularEncoderDeep

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
    
    # Tabular encoders (Fed-Heart Disease, etc.)
    "mlp_tabular": MLPTabularEncoder,
    "mlp_tabular_large": MLPTabularEncoderLarge,
    "mlp_tabular_ln": MLPTabularEncoderLN,  # LayerNorm version for extreme imbalance
    "mlp_tabular_deep": MLPTabularEncoderDeep,  # 4-layer deep encoder
}
