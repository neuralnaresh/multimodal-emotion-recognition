from dlpipeline.config.config import nested_configuration_property

@nested_configuration_property
class FusionConfig:
    hidden_size: int = 128
    dropout: float = 0.2