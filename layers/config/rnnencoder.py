from dlpipeline.config.config import nested_configuration_property

@nested_configuration_property
class RNNEncoderConfig:
    hidden_size: int = 256
    output_size: int = 64
    layers: int = 4
    dropout: float = 0.3
    bidirectional: bool = True