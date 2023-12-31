from dlpipeline.config.config import configuration_property

@configuration_property(prefix='experiment.data.transform.config.embed_audio')
class EmbedAudioConfig:
    hop_size: float = 0.96