import enum
import typing
import dataclasses

from dlpipeline.config.config import nested_configuration_property

from layers.config.fcinjection import FCInjectionConfig
from layers.config.multiinjection import MultiInjectionConfig

AttentionConfigType = typing.Union[MultiInjectionConfig, FCInjectionConfig, None]

class AttentionType(enum.Enum):
    NONE = 'none'
    INJECT = 'inject'
    MULTIINJECT = 'multiinject'

@nested_configuration_property
class AttentionBankConfig:
    @nested_configuration_property
    class AttentionModule:
        type: AttentionType
        inputs: typing.List[str]
        config: AttentionConfigType = None

    modules: typing.List[AttentionModule] = dataclasses.field(default_factory=lambda: [])