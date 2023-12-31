import enum
import typing
import dataclasses

from dlpipeline.config.config import nested_configuration_property

from losses.config.focal import FocalLossConfig
from losses.config.alfamix import ALFAMixLossConfig
from losses.config.multilabelfocal import MultiLabelFocalLossConfig
from losses.config.multilabelsoftmargin import MultiLabelSoftMarginLossConfig

LossConfigType = typing.Union[FocalLossConfig, ALFAMixLossConfig, MultiLabelFocalLossConfig, MultiLabelSoftMarginLossConfig, None] 

class LossType(enum.Enum):
    FOCAL = 'focal'
    ALFAMIX = 'alfamix'
    AVID = 'avid'
    MULTILABELFOCAL = 'multilabelfocal'
    MULTILABELSOFTMARGIN = 'multilabelsoftmargin'

@nested_configuration_property
class LossConfig:
    @nested_configuration_property
    class Loss:
        type: LossType
        stages: typing.List[int] = dataclasses.field(default_factory=list)
        results: typing.List[str] = dataclasses.field(default_factory=lambda: [-1])
        config: LossConfigType = None
        weight: float = 1

    losses: typing.List[Loss] = dataclasses.field(default_factory=list)