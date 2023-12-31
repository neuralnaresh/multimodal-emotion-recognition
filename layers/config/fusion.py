import enum
import typing
import dataclasses

from dlpipeline.config.config import nested_configuration_property

FusionConfigType = None

class FusionType(enum.Enum):
    NONE = 'none'
    ALFAMIX = 'alfamix'

class FusionSource(enum.Enum):
    ENCODER = 'encoder'
    FUSION = 'fusion'

@nested_configuration_property
class FusionConfig:
    @nested_configuration_property
    class Fusion:
        type: FusionType
        inputs: typing.Union[typing.List[str], None] = None
        config: FusionConfigType = None

    modules: typing.List[Fusion] = dataclasses.field(default_factory=lambda: [])
    source: FusionSource = FusionSource.FUSION