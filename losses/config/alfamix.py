from dlpipeline.config.config import nested_configuration_property

from losses.config.reduction import Reduction

@nested_configuration_property
class ALFAMixLossConfig:
    epsilon: float = 0.3
    reduction: Reduction = Reduction.MEAN