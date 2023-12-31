import typing
import dataclasses

from dlpipeline.config.config import nested_configuration_property

@nested_configuration_property
class STGCNEncoderConfig:
    output_size: int = 16
    temporal_kernel_size: int = 9
    channels: typing.List[int] = dataclasses.field(default_factory=lambda: [64, 64, 64, 64, 128, 128, 128, 256, 256])
    importance_weighting: bool = True
    dropout: float = 0.2