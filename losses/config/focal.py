import typing

from dlpipeline.config.config import nested_configuration_property

from losses.config.reduction import Reduction

@nested_configuration_property
class FocalLossConfig:
    gamma: float = 1.
    alpha: typing.Union[float, typing.List[float], None] = None
    reduction: Reduction = Reduction.MEAN