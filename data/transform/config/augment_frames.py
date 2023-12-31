from dlpipeline.config.config import configuration_property

@configuration_property(prefix='experiment.data.transform.config.augment_frames')
class AugmentFramesConfig:
    augmentations: int = 5