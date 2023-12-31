import enum

from dlpipeline.config.config import configuration_property

@configuration_property(prefix='experiment.data.tranform.config.load_audio')
class LoadAudioConfig:
    sample_rate: int = 48000
    samples: int = 192000