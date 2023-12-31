import typing

import torch

import data.constants

from layers.common import EncoderModuleBuild, ModalityMeta

from losses.avid import AVIDLoss
from losses.focal import FocalLoss
from losses.alfamix import ALFAMixLoss
from losses.multilabelfocal import MultiLabelFocalLoss
from losses.multilabelsoftmargin import MultiLabelSoftMarginLoss

from losses.config.focal import FocalLossConfig
from losses.config.alfamix import ALFAMixLossConfig
from losses.config.avid import AVIDLossConfig
from losses.config.multilabelfocal import MultiLabelFocalLossConfig
from losses.config.multilabelsoftmargin import MultiLabelSoftMarginLossConfig

from losses.config.loss import LossConfig, LossType

_INPUT_TYPE = typing.TypeVar("_INPUT_TYPE")


class Losses(torch.nn.Module):
    def __init__(
        self,
        config: LossConfig,
        stages_meta: typing.List[typing.Dict[str, ModalityMeta]],
        meta: typing.Dict[str, typing.Any],
    ) -> None:
        super().__init__()

        self.config = config

        self._build_losses(config, stages_meta, meta)

    def _build_losses(
        self,
        config: LossConfig,
        stages_meta: typing.List[typing.Dict[str, ModalityMeta]],
        meta: typing.Dict[str, typing.Any],
    ) -> None:
        self.losses = torch.nn.ModuleList()

        self._losses_meta: typing.List[ModalityMeta] = []

        _get_config = lambda config, type: config if config is not None else type()

        for loss in config.losses:
            if loss.type == LossType.FOCAL:
                if len(loss.stages) > 0:
                    raise ValueError(
                        f"Focal loss does not require any embedding stages"
                    )

                if len(loss.results) > 1:
                    raise ValueError(f"Focal loss does not support multiple results")

                self.losses.append(FocalLoss(_get_config(loss.config, FocalLossConfig)))

                self._losses_meta.append(
                    ModalityMeta(
                        input_constructor=lambda _, __, results: (
                            results[loss.results[0]],
                        ),
                        build=EncoderModuleBuild(size=1),
                    )
                )

            elif loss.type == LossType.MULTILABELFOCAL:
                if len(loss.stages) > 0:
                    raise ValueError(
                        f"MultiLabelFocal loss does not require any embedding stages"
                    )

                if len(loss.results) > 1:
                    raise ValueError(f"MultiLabelFocal loss does not support multiple results")

                self.losses.append(MultiLabelFocalLoss(_get_config(loss.config, MultiLabelFocalLossConfig)))

                self._losses_meta.append(
                    ModalityMeta(
                        input_constructor=lambda _, __, results: (
                            results[loss.results[0]],
                        ),
                        build=EncoderModuleBuild(size=1),
                    )
                )

            elif loss.type == LossType.MULTILABELSOFTMARGIN:
                if len(loss.stages) > 0:
                    raise ValueError(
                        f"MultiLabelSoftMargin loss does not require any embedding stages"
                    )

                if len(loss.results) > 1:
                    raise ValueError(
                        f"MultiLabelSoftMargin loss does not support multiple results"
                    )

                self.losses.append(
                    MultiLabelSoftMarginLoss(
                        #_get_config(loss.config, MultiLabelSoftMarginLossConfig)
                        #MultiLabelSoftMarginLossConfig(weight = torch.FloatTensor([6.187165, 7.924657, 12.576086, 3.453731, 10.330357, 4.059649]))
                        MultiLabelSoftMarginLossConfig(weight = torch.FloatTensor([3.288770060510738, 4.212328837809628, 6.684782427043956, 0.0, 5.491071462890625, 2.1578947561495845]))
                    )
                )

                self._losses_meta.append(
                    ModalityMeta(
                        input_constructor=lambda _, __, results: (
                            results[loss.results[0]],
                        ),
                        build=EncoderModuleBuild(size=1),
                    )
                )

            elif loss.type == LossType.ALFAMIX:
                if len(loss.stages) > 0:
                    raise ValueError(
                        f"ALFAMix loss does not require any embedding stages"
                    )

                if len(loss.results) != 2:
                    raise ValueError(
                        f"ALFAMix loss requires two results - one each from the fused and unfused features"
                    )

                self.losses.append(
                    ALFAMixLoss(_get_config(loss.config, ALFAMixLossConfig))
                )

                self._losses_meta.append(
                    ModalityMeta(
                        input_constructor=lambda _, __, results, loss=loss: (
                            results[loss.results[0]],
                            results[loss.results[1]],
                        ),
                        build=EncoderModuleBuild(size=1),
                    )
                )
            elif loss.type == LossType.AVID:
                if len(loss.stages) != 1:
                    raise ValueError("AVID loss requires exactly one embedding stage")

                if len(loss.results) != 0:
                    raise ValueError("AVID loss does not require any results")

                self.losses.append(
                    AVIDLoss(
                        meta[data.constants.META_N_TRAIN],
                        meta[data.constants.META_MAX_UTTERANCE_LENGTH]
                        if data.constants.META_MAX_UTTERANCE_LENGTH in meta
                        else 1,
                        {
                            modality: modality_meta.build.size
                            for modality, modality_meta in stages_meta[
                                loss.stages[0]
                            ].items()
                        },
                        _get_config(loss.config, AVIDLossConfig),
                    )
                )

                self._losses_meta.append(
                    ModalityMeta(
                        input_constructor=lambda input, embeddings, __, loss=loss: (
                            embeddings[loss.stages[0]],
                            input.indexes,
                        ),
                        build=EncoderModuleBuild(size=1),
                    )
                )

    def forward(
        self,
        input: _INPUT_TYPE,
        embeddings: typing.List[typing.Dict[str, torch.Tensor]],
        results: typing.List[torch.Tensor],
        labels: torch.Tensor,
        utterance_lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        total_loss = torch.zeros(1, device=labels.device)

        for i, loss in enumerate(self.losses):
            total_loss += self.config.losses[i].weight * loss(
                *self._losses_meta[i].input_constructor(input, embeddings, results),
                labels,
                utterance_lengths,
            )

        return total_loss
