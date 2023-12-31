import typing

from dlpipeline.config.config import nested_configuration_property

@nested_configuration_property
class SpatioTemporalPositionEncoderConfig:
    hidden_size: int = 1024
    max_temporal_buckets: int = 8
    max_vertical_buckets: int = 14
    max_horizontal_buckets: int = 14
    dropout: float = 0.1
    epsilon: float = 1e-6