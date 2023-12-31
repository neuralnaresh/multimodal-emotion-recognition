from dlpipeline.config.config import nested_configuration_property

@nested_configuration_property
class AVIDMemoryConfig:
    inter: bool = True
    intra: bool = True
    negatives: int = 1024
    momentum: float = 0.9
    temperature: float = 0.07

@nested_configuration_property
class AVIDLossConfig:
    memory: AVIDMemoryConfig = AVIDMemoryConfig()
    intra: float = 1.0
    inter: float = 1.0