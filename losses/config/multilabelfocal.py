import dataclasses
import typing

from dlpipeline.config.config import nested_configuration_property

from losses.config.reduction import Reduction

@nested_configuration_property
class MultiLabelFocalLossConfig:
    gamma: float = 2.0
    alpha: typing.Union[float, typing.List[float], None] = dataclasses.field(default_factory=lambda: [0.5397482038969184, 0.6899094269349753, 1.0919098400464096, 0.3035204546050344, 0.8978218330364319, 0.35588339969641564])
    reduction: Reduction = Reduction.MEAN