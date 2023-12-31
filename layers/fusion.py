import typing

import torch

from layers.common import EncoderModuleBuild, ModalityMeta

from layers.alfamix import ALFAMixFusion

from layers.config.fusion import FusionConfig, FusionType


class Fusion(torch.nn.Module):
    def __init__(
        self, config: FusionConfig, encoders: typing.Dict[str, ModalityMeta]
    ) -> None:
        super().__init__()

        self._build_fusion(config, encoders)

    def _build_fusion(
        self, config: FusionConfig, encoders: typing.Dict[str, ModalityMeta]
    ) -> None:
        self.fusion_modules = torch.nn.ModuleDict()

        self._fusion_meta: typing.Dict[str, ModalityMeta] = {}

        _get_config = lambda config, type: config if config is not None else type()

        for fusion in [f for f in config.modules if f.type is not FusionType.NONE]:
            if fusion.inputs is None or len(fusion.inputs) == 0:
                fusion.inputs = encoders.keys()

            if fusion.type == FusionType.ALFAMIX:
                input_sizes = list(set(
                    [encoders[input].build.size for input in fusion.inputs]
                ))

                if len(input_sizes) > 1:
                    raise ValueError(
                        f"ALFAMix fusion only supports inputs with the same size"
                    )

                name = "_".join(fusion.inputs)

                self.fusion_modules[name] = ALFAMixFusion(
                    input_sizes[0], len(fusion.inputs)
                )

                self._fusion_meta[name] = ModalityMeta(
                    input_constructor=lambda encodings: {
                        input: encodings[input] for input in fusion.inputs
                    },
                    build=EncoderModuleBuild(size=len(fusion.inputs) * input_sizes[0]),
                )

    def forward(
        self, encodings: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, torch.Tensor]:
        outputs = encodings

        for name, fusion in self.fusion_modules.items():
            output = fusion(
                self._fusion_meta[name].input_constructor(encodings)
            )

            if isinstance(output, dict):
                outputs.update(output)
            else:
                outputs[name] = output

        return outputs
