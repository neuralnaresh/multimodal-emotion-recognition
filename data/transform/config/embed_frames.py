import enum

from dlpipeline.config.config import configuration_property

class EmbedFramesSource(enum.Enum):
    FRAMES = 'frames'
    FACES = 'faces'
    BG = 'bg'

@configuration_property(prefix='experiment.data.transform.config.embed_frames')
class EmbedFramesConfig:
    source: EmbedFramesSource = EmbedFramesSource.FACES
    batch_size: int = 16