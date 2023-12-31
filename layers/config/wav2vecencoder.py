import enum

from dlpipeline.config.config import nested_configuration_property

class Pooling(enum.Enum):
    MEAN = 'mean'
    MAX = 'max'
    SUM = 'sum'

@nested_configuration_property
class Wav2VecEncoderConfig:
    output_size: int = 256
    dropout: float = 0.3
    pooling: Pooling = Pooling.MEAN