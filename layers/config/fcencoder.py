from dlpipeline.config.config import nested_configuration_property

@nested_configuration_property
class FCEncoderConfig:
    hidden_size: int = 16
    output_size: int = 16
    layers: int = 4
    dropout: float = 0.3
