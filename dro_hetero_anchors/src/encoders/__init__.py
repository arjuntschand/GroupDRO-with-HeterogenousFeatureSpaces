"""Encoder registry.

Configs name an encoder by string; training loops look it up here. The
image/text encoders used in early exploration (MNIST/USPS CNNs, TextCaps
visual/text encoders, the DICOM multi-view mammography encoder) live under
legacy/ and are not registered. The EMBED results in the paper use the
frozen-ViT pipeline in train_embed_xenia.py, which builds its model directly.
"""
from .tabular_encoder import (
    MLPTabularEncoder,
    MLPTabularEncoderLarge,
    MLPTabularEncoderLN,
    MLPTabularEncoderDeep,
)

ENCODER_REGISTRY = {
    # Tabular encoders (Fed-Heart Disease, NHANES)
    "mlp_tabular": MLPTabularEncoder,
    "mlp_tabular_large": MLPTabularEncoderLarge,
    "mlp_tabular_ln": MLPTabularEncoderLN,  # LayerNorm version for extreme imbalance
    "mlp_tabular_deep": MLPTabularEncoderDeep,  # 4-layer deep encoder
}
