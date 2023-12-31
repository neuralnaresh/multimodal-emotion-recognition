import typing
import dataclasses

import torch

from layers.common import ModalityMeta, EncoderModuleBuild

from layers.fcinjection import FCInjection
from layers.multiinjection import MultiInjection

from layers.config.fcinjection import FCInjectionConfig
from layers.config.multiinjection import MultiInjectionConfig
from layers.config.attentionbank import (
    AttentionBankConfig,
    AttentionType,
)

_INPUT_TYPE = typing.TypeVar("_INPUT_TYPE")


class AttentionBank(torch.nn.Module):
    def __init__(
        self,
        input_type: typing.Type,
        config: AttentionBankConfig,
        build: typing.Dict[str, EncoderModuleBuild],
        names: typing.Dict[str, str] = {},
        lengths: typing.Dict[str, str] = {},
    ) -> None:
        super().__init__()

        self._build_attention_modules(input_type, config, build, names, lengths)

    def _build_attention_modules(
        self,
        input_type: typing.Type,
        config: AttentionBankConfig,
        build: typing.Dict[str, EncoderModuleBuild],
        names: typing.Dict[str, str],
        lengths: typing.Dict[str, str],
    ) -> None:
        self.attention_modules = torch.nn.ModuleDict()

        self._attention_meta: typing.Dict[str, ModalityMeta] = {}

        fields: typing.List[dataclasses.Field] = list(
            getattr(input_type, "__dataclass_fields__", {}).values()
        )

        for attention in [
            a for a in config.modules if a.type is not AttentionType.NONE
        ]:
            constructors = []
            input_fields = []

            for input in attention.inputs:
                field = next((f for f in fields if f.name == input), None)

                if field is None and input not in names:
                    raise ValueError(
                        f"Field {input} not found in input type {input_type}"
                    )
                elif field is None and input in names:
                    field = next((f for f in fields if f.name == names[input]), None)

                if field is None:
                    raise ValueError(
                        f"Field {input} not found in input type {input_type}"
                    )

                input_fields.append(field)

                _build_input_constructor = (
                    lambda length, input=input, field=field: (
                        lambda x: (
                            getattr(x, field.name),
                            getattr(
                                x,
                                lengths[input]
                                if input in lengths
                                else f"{field.name}_seq_length",
                            ),
                        )
                    )
                    if length
                    else lambda x: (getattr(x, field.name),)
                )

                if attention.type == AttentionType.INJECT:
                    constructors.append(_build_input_constructor)

            _build_meta = (
                lambda config, ordering, length=False, dialog=False, build=None, constructors=constructors: ModalityMeta(
                    lambda x: sum([constructors[index](length)(x) for index in ordering], ()),
                    EncoderModuleBuild(size=config.output_size) if build is None else build,
                    dialog,
                )
            )

            _get_config = lambda config, type: config if config is not None else type()

            if attention.type == AttentionType.INJECT:
                if len(input_fields) != 2:
                    raise ValueError("Injection must have two inputs")

                injection_payload = input_fields[0].name
                injection_target = input_fields[1].name

                config = _get_config(attention.config, FCInjectionConfig)

                self.attention_modules[injection_target] = FCInjection(
                    build[injection_target].size,
                    build[injection_payload].size,
                    config,
                )

                self._attention_meta[injection_target] = _build_meta(config, [1, 0])

            elif attention.type == AttentionType.MULTIINJECT:
                if not len(input_fields) >= 2:
                    raise ValueError("MultiInjection must have at least two inputs")

                injection_payloads = [field.name for field in input_fields[:-1]]
                injection_target = input_fields[-1].name

                config = _get_config(attention.config, MultiInjectionConfig)

                self.attention_modules[injection_target] = MultiInjection(
                    build[injection_target].size,
                    [build[payload].size for payload in injection_payloads],
                    config
                )

                self._attention_meta[injection_target] = ModalityMeta(
                    lambda x, injection_target=injection_target, injection_payloads=injection_payloads: (getattr(x, injection_target), [getattr(x, payload) for payload in injection_payloads]),
                    EncoderModuleBuild(size=config.output_size),
                    dialog=False
                )


    def forward(
        self, input: _INPUT_TYPE, utterance_lengths: torch.Tensor = None
    ) -> typing.Dict[str, torch.Tensor]:
        outputs = {}

        for name, attention in self.attention_modules.items():
            if self._attention_meta[name].dialog:
                outputs[name] = attention(
                    *(self._attention_meta[name].input_constructor(input)),
                    utterance_lengths,
                )
            else:
                outputs[name] = attention(
                    *self._attention_meta[name].input_constructor(input)
                )

        return outputs
