import enum
import typing
import dataclasses

from dlpipeline.config.config import nested_configuration_property

from layers.config.fcencoder import FCEncoderConfig
from layers.config.rnnencoder import RNNEncoderConfig
from layers.config.drnnencoder import DRNNEncoderConfig
from layers.config.stgcnencoder import STGCNEncoderConfig
from layers.config.wav2vecencoder import Wav2VecEncoderConfig
from layers.config.graphormerencoder import GraphormerEncoderConfig
from layers.config.timesformerencoder import TimesformerEncoderConfig
from layers.config.mwmsg3dencoder import MultiWindowMSG3DEncoderConfig

EncoderConfigType = typing.Union[
    FCEncoderConfig,
    RNNEncoderConfig,
    DRNNEncoderConfig,
    MultiWindowMSG3DEncoderConfig,
    STGCNEncoderConfig,
    Wav2VecEncoderConfig,
    TimesformerEncoderConfig,
    GraphormerEncoderConfig,
    None,
]


class EncoderType(enum.Enum):
    NONE = "none"
    FC = "fc"
    RNN = "rnn"
    DRNN = "drnn"
    MWMSG = "mwmsg"
    DMWMSG = "dmwmsg"
    DSTGCN = "dstgcn"
    TIMESFORMER = "timesformer"
    WAV2VEC = "wav2vec"
    GRAPHORMER = "graphormer"

@nested_configuration_property
class DialogEncoderBankConfig:
    @nested_configuration_property
    class Encoder:
        name: str
        type: EncoderType
        config: EncoderConfigType = None

    encoders: typing.List[Encoder] = dataclasses.field(default_factory=lambda: [])
