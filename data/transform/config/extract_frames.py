from dlpipeline.config.config import configuration_property

@configuration_property(prefix='experiment.data.tranform.config.extract_frames')
class ExtractFramesConfig:
    frame_rate: int = 10