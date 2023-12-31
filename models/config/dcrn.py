import enum
import typing
import dataclasses

from dlpipeline.config.config import configuration_property, nested_configuration_property

from layers.config.attentionbank import AttentionBankConfig
from layers.config.dencoderbank import DialogEncoderBankConfig
from layers.config.fusion import FusionConfig
from losses.config.loss import LossConfig

from layers.config.dcrncognition import DCRNCognitionConfig

class DCRNBase(enum.Enum):
    LSTM = 'lstm'
    GRU = 'gru'
    LINEAR = 'linear'

@configuration_property(prefix='experiment.model.config.dcrn')
class DCRNConfig:
    @nested_configuration_property
    class FocalLoss:
        gamma: float = 1
        alpha: typing.Union[float, None] = None

        average: bool = True

    base: DCRNBase = DCRNBase.LSTM

    layers: int = 2
    hidden: int = 100
    dropout: float = 0.2

    attention: AttentionBankConfig = AttentionBankConfig()
    encoders: DialogEncoderBankConfig = DialogEncoderBankConfig()
    fusions: typing.List[FusionConfig] = dataclasses.field(default_factory=lambda: [FusionConfig()])
    losses: LossConfig = LossConfig()

    classify: typing.List[int] = dataclasses.field(default_factory=lambda: [-1])

    cognition: DCRNCognitionConfig = DCRNCognitionConfig()