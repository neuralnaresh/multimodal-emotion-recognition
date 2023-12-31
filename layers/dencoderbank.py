import typing
import dataclasses

import torch

from layers.common import ModalityMeta, EncoderModuleBuild

from layers.fcencoder import FCEncoder
from layers.rnnencoder import RNNEncoder
from layers.drnnencoder import DRNNEncoder
from layers.dstgcnencoder import DialogSTGCNEncoder
from layers.dmsg3d import DialogMultiWindowMSG3DEncoder
from layers.wav2vecencoder import Wav2VecEncoder
from layers.graphormerencoder import GraphormerEncoder
from layers.timesformerencoder import TimesformerEncoder
from layers.mwmsg3dencoder import MultiWindowMSG3dEncoder

from layers.config.fcencoder import FCEncoderConfig
from layers.config.rnnencoder import RNNEncoderConfig
from layers.config.drnnencoder import DRNNEncoderConfig
from layers.config.stgcnencoder import STGCNEncoderConfig
from layers.config.wav2vecencoder import Wav2VecEncoderConfig
from layers.config.graphormerencoder import GraphormerEncoderConfig
from layers.config.timesformerencoder import TimesformerEncoderConfig
from layers.config.mwmsg3dencoder import MultiWindowMSG3DEncoderConfig
from layers.config.dencoderbank import DialogEncoderBankConfig, EncoderType

_INPUT_TYPE = typing.TypeVar("_INPUT_TYPE")


class DialogEncoderBank(torch.nn.Module):
    def __init__(
        self,
        input_type: typing.Type,
        config: DialogEncoderBankConfig,
        attention: typing.Dict[str, ModalityMeta],
        build: typing.Dict[str, EncoderModuleBuild],
        names: typing.Dict[str, str] = {},
        lengths: typing.Dict[str, str] = {},
    ) -> None:
        super().__init__()

        self._build_encoders(input_type, config, attention, build, names, lengths)

    @property
    def output_shapes(self) -> typing.Dict[str, int]:
        return {
            name: encoder.output_size for name, encoder in self._encoder_meta.items()
        }

    def _build_encoders(
        self,
        input: typing.Type,
        config: DialogEncoderBankConfig,
        attention: typing.Dict[str, ModalityMeta],
        build: typing.Dict[str, EncoderModuleBuild],
        names: typing.Dict[str, str],
        lengths: typing.Dict[str, str],
    ) -> torch.nn.ModuleDict:
        self.encoder_modules = torch.nn.ModuleDict()

        self._encoder_meta: typing.Dict[str, ModalityMeta] = {}

        fields: typing.List[dataclasses.Field] = list(
            getattr(input, "__dataclass_fields__", {}).values()
        )

        for encoder in [e for e in config.encoders if e.type != EncoderType.NONE]:
            if encoder.name in attention:
                name = encoder.name
                current_build = attention[encoder.name].build
                from_attention = True
            else:
                field = next((f for f in fields if f.name == encoder.name), None)

                if field is None and encoder.name not in names:
                    raise ValueError(
                        f"Field {encoder.name} not found in input type {input}"
                    )
                elif field is None and encoder.name in names:
                    field = next(
                        (f for f in fields if f.name == names[encoder.name]), None
                    )

                if field is None:
                    raise ValueError(
                        f"Field {encoder.name} not found in input type {input}"
                    )

                name = field.name
                current_build = build[field.name]
                from_attention = False

            _get_config = lambda config, type: config if config is not None else type()

            _build_input_constructor = (
                lambda length, name=name, from_attention=from_attention: (
                    lambda input, attention: (
                        getattr(input, name) if not from_attention else attention[name],
                        getattr(
                            input,
                            lengths[name] if name in lengths else f"{name}_seq_length",
                        ),
                    )
                )
                if length
                else lambda input, attention: (
                    getattr(input, name) if not from_attention else attention[name],
                )
            )

            _build_meta = lambda current_config, length=False, dialog=False, build=None: ModalityMeta(
                _build_input_constructor(length),
                EncoderModuleBuild(size=current_config.output_size)
                if build is None
                else build,
                dialog,
            )

            if encoder.type == EncoderType.FC:
                current_config = _get_config(encoder.config, FCEncoderConfig)

                self.encoder_modules[name] = FCEncoder(
                    current_build.size,
                    current_config,
                )
                self._encoder_meta[name] = _build_meta(current_config)
            elif encoder.type == EncoderType.RNN:
                current_config = _get_config(encoder.config, RNNEncoderConfig)

                self.encoder_modules[name] = RNNEncoder(
                    current_build.size,
                    current_config,
                )
                self._encoder_meta[name] = _build_meta(current_config, length=True)
            elif encoder.type == EncoderType.DRNN:
                current_config = _get_config(encoder.config, DRNNEncoderConfig)

                self.encoder_modules[name] = DRNNEncoder(
                    current_build.size,
                    current_config,
                )
                self._encoder_meta[name] = _build_meta(
                    current_config, length=True, dialog=True
                )
            elif encoder.type == EncoderType.MWMSG:
                current_config = MultiWindowMSG3DEncoderConfig(output_size=64)

                self.encoder_modules[name] = MultiWindowMSG3dEncoder(
                    current_build.channels,
                    current_build.vertices,
                    current_build.persons,
                    current_config,
                    current_build.adjacency_matrix,
                )
                self._encoder_meta[name] = _build_meta(current_config)
            elif encoder.type == EncoderType.DMWMSG:
                current_config = _get_config(
                    encoder.config, MultiWindowMSG3DEncoderConfig
                )

                self.encoder_modules[name] = DialogMultiWindowMSG3DEncoder(
                    current_build.channels,
                    current_build.vertices,
                    current_build.persons,
                    current_config,
                    current_build.adjacency_matrix,
                )
                self._encoder_meta[name] = _build_meta(current_config, dialog=True)
            elif encoder.type == EncoderType.DSTGCN:
                current_config = _get_config(
                    encoder.config, STGCNEncoderConfig
                )

                self.encoder_modules[name] = DialogSTGCNEncoder(
                    current_build.vertices,
                    current_build.channels,
                    current_build.adjacency_matrix,
                    current_config,
                )
                self._encoder_meta[name] = _build_meta(current_config, dialog=True)
            elif encoder.type == EncoderType.TIMESFORMER:
                current_config = _get_config(encoder.config, TimesformerEncoderConfig)

                self.encoder_modules[name] = TimesformerEncoder(current_config)
                self._encoder_meta[name] = _build_meta(current_config, dialog=False, length=True)
            elif encoder.type == EncoderType.WAV2VEC:
                current_config = _get_config(encoder.config, Wav2VecEncoderConfig)

                self.encoder_modules[name] = Wav2VecEncoder(current_config)
                self._encoder_meta[name] = _build_meta(current_config, dialog=False, length=True)
            elif encoder.type == EncoderType.GRAPHORMER:
                current_config = _get_config(encoder.config, GraphormerEncoderConfig)

                self.encoder_modules[name] = GraphormerEncoder(current_config)
                self._encoder_meta[name] = _build_meta(current_config, dialog=False, length=False)

    def forward(
        self,
        input: _INPUT_TYPE,
        attention: typing.Dict[str, typing.Any],
        utterance_lengths: torch.Tensor = None,
    ) -> typing.Dict[str, torch.Tensor]:
        outputs = attention

        for name, encoder in self.encoder_modules.items():
            if self._encoder_meta[name].dialog:
                outputs[name] = encoder(
                    *(self._encoder_meta[name].input_constructor(input, attention)),
                    utterance_lengths,
                )
            else:
                outputs[name] = encoder(
                    *self._encoder_meta[name].input_constructor(input, attention)
                )
        return outputs
