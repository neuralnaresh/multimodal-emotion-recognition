from dlpipeline.config.config import nested_configuration_property

@nested_configuration_property
class GraphormerEncoderConfig:
    output_size: int = 256
    dropout: float = 0.3
    spatial_pox_max: int = 20
