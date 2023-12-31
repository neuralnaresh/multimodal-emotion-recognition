import enum

from dlpipeline.config.config import configuration_property

class FramesSource(enum.Enum):
    FRAME = 'frame'
    FACE = 'face'
    BG = 'bg'

@configuration_property(prefix='experiment.data.tranform.config.load_frames')
class LoadFramesConfig:
    source: FramesSource = FramesSource.FACE
    frames: int = 16