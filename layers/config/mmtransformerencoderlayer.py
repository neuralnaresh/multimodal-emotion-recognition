import enum

from dlpipeline.config.config import nested_configuration_property

from layers.attentionfusion import AttentionFusionMechanism

class Activation(enum.Enum):
    RELU = 'relu'
    GELU = 'gelu'
    SWISH = 'swish'
    GEGLU = 'geglu'

@nested_configuration_property
class MultiModalTransformerEncoderLayerConfig:
    @nested_configuration_property
    class FusionConfig:
        pre_forward: bool = True
        mechanism: AttentionFusionMechanism = AttentionFusionMechanism.HADAMARD
        normalize: bool = True

    heads: int = 16
    key_value_size: int = 64
    feed_forward_size: int = 4096
    pre_normalize: bool = False
    bias: bool = False
    activation: Activation = Activation.GELU
    dropout: float = 0.1
    epsilon: float = 1e-6

    fusion: FusionConfig = FusionConfig()