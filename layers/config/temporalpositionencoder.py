import typing

from dlpipeline.config.config import nested_configuration_property

@nested_configuration_property
class TemporalPositionEncoderConfig:
    hidden_size: int = 1024
    max_temporal_buckets: int = 1200
    dropout: float = 0.1
    initializer_range: typing.Union[float, None] = None
    epsilon: float = 1e-6