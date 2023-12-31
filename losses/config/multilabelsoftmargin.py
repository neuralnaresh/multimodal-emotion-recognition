import typing

from dlpipeline.config.config import nested_configuration_property

from losses.config.reduction import Reduction

@nested_configuration_property
class MultiLabelSoftMarginLossConfig:
    weight: typing.Optional[typing.List[float]] = None
    reduction: Reduction = Reduction.SUM