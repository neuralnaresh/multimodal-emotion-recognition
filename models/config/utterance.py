import typing
import dataclasses

from dlpipeline.config.config import configuration_property

from layers.config.dencoderbank import DialogEncoderBankConfig
from layers.config.attentionbank import AttentionBankConfig
from layers.config.fcencoder import FCEncoderConfig
from layers.config.fusion import FusionConfig
from losses.config.loss import LossConfig

@configuration_property(prefix='experiment.model.config.utterance')
class UtteranceConfig:
    attention: AttentionBankConfig = AttentionBankConfig()
    encoders: DialogEncoderBankConfig = DialogEncoderBankConfig()
    fusions: typing.List[FusionConfig] = dataclasses.field(default_factory=lambda: [])
    losses: LossConfig = LossConfig()

    classify: typing.List[int] = dataclasses.field(default_factory=lambda: [-1])

    classifier: FCEncoderConfig = FCEncoderConfig()