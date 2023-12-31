from dlpipeline.config.config import nested_configuration_property

@nested_configuration_property
class FCInjectionConfig:
    hidden_size: int = 16
    output_size: int = 16
    layers: int = 4
    heads: int = 8
    dropout: float = 0.4