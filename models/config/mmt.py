from dlpipeline.config.config import configuration_property, nested_configuration_property

from layers.config.mmtransformerencoderlayer import MultiModalTransformerEncoderLayerConfig
from layers.config.spatiotemporalpositionencoder import SpatioTemporalPositionEncoderConfig
from layers.config.temporalpositionencoder import TemporalPositionEncoderConfig

@configuration_property(prefix='experiment.model.config.mmt')
class MultiModalTransformerModelConfig:
    @nested_configuration_property
    class Patches:
        video_temporal: int = 4
        video_spatial: int = 16
        audio_temporal: int = 128
        text_embedding: int = 512
    
    patches: Patches = Patches()

    text_position: TemporalPositionEncoderConfig = TemporalPositionEncoderConfig()
    audio_position: TemporalPositionEncoderConfig = TemporalPositionEncoderConfig()
    video_position: SpatioTemporalPositionEncoderConfig = SpatioTemporalPositionEncoderConfig()
    
    encoder: MultiModalTransformerEncoderLayerConfig = MultiModalTransformerEncoderLayerConfig()

    layers: int = 2
    layer_output_size: int = 1024

    post_projection_size: int = 1024
    classification_hidden_size: int = 512

    calibration_weight: float = 1.0
    nce_weight: float = 1.0
    kd_weight: float = 1.0