import typing

from dlpipeline.config.config import nested_configuration_property

@nested_configuration_property
class MultiInjectionConfig:
    hidden_size: int = 16
    output_size: int = 16
    layers: int = 4
    heads: int = 8
    dropout: float = 0.4

    payload_key_size: int = 512
    payload_key_layers: int = 2
    payload_value_size: int = 512
    payload_value_layers: int = 2
    payload_output_size: int = 256
    payload_output_layers: int = 1