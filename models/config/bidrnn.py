from dlpipeline.config.config import configuration_property, nested_configuration_property

from layers.config.fcencoder import FCEncoderConfig
from layers.config.drnnencoder import DRNNEncoderConfig
from layers.config.mwmsg3dencoder import MultiWindowMSG3DEncoderConfig
from layers.config.simplefusion import FusionConfig

@configuration_property(prefix='experiment.model.config.bidrnn')
class BiDRNNConfig:
    @nested_configuration_property
    class DRNNEncodersList:
        text: bool = True
        frames: bool = True
        audio: bool = True
        audio_features: bool = False
        primary_hog: bool = False
        secondary_hog: bool = False
        primary_landmarks: bool = False
        secondary_landmarks: bool = False
        primary_aus: bool = False
        secondary_aus: bool = False

    encoders: DRNNEncodersList = DRNNEncodersList()

    text: FCEncoderConfig = FCEncoderConfig()
    frames: DRNNEncoderConfig = DRNNEncoderConfig()
    audio: DRNNEncoderConfig = DRNNEncoderConfig()
    audio_features: FCEncoderConfig = FCEncoderConfig()
    primary_hog: DRNNEncoderConfig = DRNNEncoderConfig()
    secondary_hog: DRNNEncoderConfig = DRNNEncoderConfig()
    primary_landmarks: MultiWindowMSG3DEncoderConfig = MultiWindowMSG3DEncoderConfig()
    secondary_landmarks: MultiWindowMSG3DEncoderConfig = MultiWindowMSG3DEncoderConfig()
    primary_aus: DRNNEncoderConfig = DRNNEncoderConfig()
    secondary_aus: DRNNEncoderConfig = DRNNEncoderConfig()

    fusion: FusionConfig = FusionConfig()

    cell_global_state_dim: int = 150
    cell_participant_state_dim: int = 150
    cell_hidden_dim: int = 100
    cell_output_dim: int = 100
    cell_dropout: float = 0.1

    emotion_attention: bool = False