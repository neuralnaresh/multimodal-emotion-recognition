from dlpipeline.config.config import configuration_property

@configuration_property(prefix='experiment.data.tranform.config.load_video')
class LoadVideoConfig:
    frames: int = 32
    
    resize: bool = True
    resize_width: int = 640
    resize_height: int = 360