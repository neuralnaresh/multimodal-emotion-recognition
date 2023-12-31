import typing
import itertools

import torch

import sklearn.metrics

import numpy as np

from dlpipeline.model.model import Model
from dlpipeline.model.input import model_input, ModelInputConfig

import data.constants

from layers.doublelinear import DoubleLinear
from layers.mmtransformerencoderlayer import MultiModalTransformerEncoderLayer
from layers.spatiotemporalpositionencoder import SpatioTemporalPositionEncoder
from layers.temporalpositionencoder import TemporalPositionEncoder

from layers.config.mmtransformerencoderlayer import Activation

from models.config.mmt import MultiModalTransformerModelConfig


class MMTOutputUnit(typing.NamedTuple):
    modalities: dict[str, torch.Tensor]
    fusion: torch.Tensor


class MultiModalTransformerModel(Model):
    @model_input({data.constants.DATA_TEXT_TOKENS: ModelInputConfig(dtype=torch.long)})
    class MMTInput:
        text_tokens: torch.Tensor  # (batch, words)
        video_raw: torch.Tensor  # (batch, channels, frames, height, width)
        audio_raw: torch.Tensor  # (batch, samples)
        labels: torch.Tensor  # (batch, classes)

    class MMTOutput(typing.NamedTuple):
        pooled: MMTOutputUnit
        projections: MMTOutputUnit
        predictions: MMTOutputUnit

    class MMTMetricRepresentation(typing.NamedTuple):
        labels: np.ndarray
        predictions: np.ndarray

    _INPUT_TYPE = MMTInput
    _OUTPUT_TYPE = MMTOutput
    _METRIC_REPRESENTATION = MMTMetricRepresentation

    def __init__(self, metadata: dict[str, typing.Any]) -> None:
        super().__init__()

        self.metadata = metadata
        self.config = MultiModalTransformerModelConfig()

        self.embeddings = torch.nn.ModuleDict(
            {
                "video": torch.nn.Conv3d(
                    3,
                    self.config.layer_output_size,
                    kernel_size=(
                        self.config.patches.video_temporal,
                        self.config.patches.video_spatial,
                        self.config.patches.video_spatial,
                    ),
                    stride=(
                        self.config.patches.video_temporal,
                        self.config.patches.video_spatial,
                        self.config.patches.video_spatial,
                    ),
                    padding="valid",
                ),
                "audio": torch.nn.Conv1d(
                    1,
                    self.config.layer_output_size,
                    kernel_size=self.config.patches.audio_temporal,
                    stride=self.config.patches.audio_temporal,
                    padding="valid",
                ),
                "text": torch.nn.Embedding(
                    2**16,
                    self.config.patches.text_embedding,
                ),
            }
        )

        self.pre_projection = torch.nn.ModuleDict(
            {
                "video": torch.nn.Linear(
                    self.config.layer_output_size, self.config.layer_output_size
                ),
                "audio": torch.nn.Linear(
                    self.config.layer_output_size, self.config.layer_output_size
                ),
                "text": torch.nn.Linear(
                    self.config.patches.text_embedding, self.config.layer_output_size
                ),
            }
        )

        self.positional_encoders = torch.nn.ModuleDict(
            {
                "video": SpatioTemporalPositionEncoder(self.config.video_position),
                "audio": TemporalPositionEncoder(self.config.audio_position),
                "text": TemporalPositionEncoder(self.config.text_position),
            }
        )

        self.encoder = MultiModalTransformerEncoderLayer(
            self.config.layer_output_size,
            self.config.layer_output_size,
            3,
            self.config.encoder,
        )

        self.post_projection = torch.nn.ModuleDict(
            {
                "video": torch.nn.Linear(
                    self.config.layer_output_size, self.config.post_projection_size
                ),
                "audio": torch.nn.Linear(
                    self.config.layer_output_size, self.config.post_projection_size
                ),
                "text": torch.nn.Linear(
                    self.config.layer_output_size, self.config.post_projection_size
                ),
                "fusion": torch.nn.Linear(
                    self.config.layer_output_size, self.config.post_projection_size
                ),
            }
        )

        self.mlp = torch.nn.ModuleDict(
            {
                "video": DoubleLinear(
                    self.config.layer_output_size,
                    self.config.classification_hidden_size,
                    metadata[data.constants.META_N_CLASSES],
                    activation=Activation.GELU,
                    norm=True,
                ),
                "audio": DoubleLinear(
                    self.config.layer_output_size,
                    self.config.classification_hidden_size,
                    metadata[data.constants.META_N_CLASSES],
                    activation=Activation.GELU,
                    norm=True,
                ),
                "text": DoubleLinear(
                    self.config.layer_output_size,
                    self.config.classification_hidden_size,
                    metadata[data.constants.META_N_CLASSES],
                    activation=Activation.GELU,
                    norm=True,
                ),
                "fusion": DoubleLinear(
                    self.config.layer_output_size,
                    self.config.classification_hidden_size,
                    metadata[data.constants.META_N_CLASSES],
                    activation=Activation.GELU,
                    norm=True,
                ),
            }
        )

        self._aggregation_tokens = torch.nn.ParameterDict(
            {
                "video": torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(1, self.config.layer_output_size)
                    )
                ),
                "audio": torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(1, self.config.layer_output_size)
                    )
                ),
                "text": torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(1, self.config.layer_output_size)
                    )
                ),
                "fusion": torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(1, self.config.layer_output_size)
                    )
                ),
            }
        )

        self._nce_temperature = torch.nn.Parameter(torch.tensor(0.08))
        self._kd_temperature = torch.nn.Parameter(torch.tensor(0.08))

    def _run_modality_layer(
        self, layer: torch.ModuleDict, inputs: list[torch.Tensor], **kwargs
    ) -> list[torch.Tensor]:
        modalities = ["video", "audio", "text"]

        outputs = []

        for i, modality in enumerate(modalities):
            if modality in layer:
                outputs.append(
                    layer[modality](
                        inputs[i], **{arg: kwargs[arg][i] for arg in kwargs}
                    )
                )

        if len(inputs) == 4:
            f = layer["fusion"](inputs[3])
            outputs.append(f)

        return outputs

    def _flatten(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        shapes = [tensor.shape for tensor in tensors]
        return [
            tensor.reshape(tensor.shape[0], -1, tensor.shape[-1]) for tensor in tensors
        ], shapes

    def _add_aggregtion_token(self, input: list[torch.Tensor]) -> list[torch.Tensor]:
        modalities = ["video", "audio", "text"]

        output = []

        for i, tokens in enumerate(input):
            output.append(
                torch.cat(
                    [
                        torch.stack(
                            [self._aggregation_tokens[modalities[i]]] * tokens.shape[0],
                            dim=0,
                        ),
                        tokens,
                    ],
                    dim=1,
                )
            )

        output.append(
            torch.stack([self._aggregation_tokens["fusion"]] * input[0].shape[0], dim=0)
        )

        return output

    def _modalities_to_unit(self, tensors: list[torch.Tensor]) -> MMTOutputUnit:
        return MMTOutputUnit(
            modalities={"video": tensors[0], "audio": tensors[1], "text": tensors[2]},
            fusion=tensors[3],
        )

    def forward(self, input: MMTInput) -> MMTOutput:
        inputs = [input.video_raw, input.audio_raw.permute(0, 2, 1), input.text_tokens]

        x = self._run_modality_layer(self.embeddings, inputs)

        x[0] = x[0].permute(0, 2, 3, 4, 1)
        x[1] = x[1].permute(0, 2, 1)

        x = self._run_modality_layer(self.pre_projection, x)

        x, shapes = self._flatten(x)
        x = self._run_modality_layer(self.positional_encoders, x, dimensions=shapes)

        for i in x:
            print(i.shape)

        x = self._add_aggregtion_token(x)

        for i in x:
            print(i.shape)

        text_mask = torch.cat(
            [
                torch.where(input.text_tokens == 0, 0.0, 1.0),
                torch.ones(input.text_tokens.shape[0], 1).to(input.text_tokens.device),
            ],
            dim=-1,
        )[:, None, None, :].to(torch.float32)
        text_mask = (1.0 - text_mask) * 1e-9

        for _ in range(self.config.layers):
            encodings = self.encoder(x, attention_mask=[None, None, text_mask, None])
            x = [encoding["hidden"] for encoding in encodings]

        pooled = [hidden[:, -1, :] for hidden in x]

        projections = self._run_modality_layer(self.post_projection, pooled)
        predictions = self._run_modality_layer(self.mlp, pooled)

        return MultiModalTransformerModel.MMTOutput(
            pooled=self._modalities_to_unit(pooled),
            projections=self._modalities_to_unit(projections),
            predictions=self._modalities_to_unit(predictions),
        )

    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = torch.nn.functional.normalize(a, dim=1)
        b = torch.nn.functional.normalize(b, dim=1)

        logits = a @ b.T * torch.exp(self._nce_temperature)
        labels = torch.arange(logits.shape[0]).to(logits.device)

        loss_a = torch.nn.functional.cross_entropy(logits, labels)
        loss_b = torch.nn.functional.cross_entropy(logits, labels)

        return loss_a + loss_b

    def _nce_loss(self, projections: dict[str, torch.Tensor]) -> torch.Tensor:
        losses = []

        for a, b in itertools.combinations(projections.keys(), 2):
            losses.append(self._cosine_similarity(projections[a], projections[b]))

        return torch.stack(losses).mean()

    def _knowledge_distillation_loss(self, modalities: list[torch.Tensor], fusion: torch.Tensor) -> torch.Tensor:
        target = torch.prod(torch.stack(modalities, dim=0), dim=0)

        return torch.nn.functional.kl_div(
            (fusion / self._kd_temperature).sigmoid().log(),
            (target / self._kd_temperature).sigmoid(),
            reduce='sum'
        )

    def _classification_loss(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return (
            torch.nn.functional.binary_cross_entropy_with_logits(
                predictions.sigmoid(), labels, reduction="sum"
            )
            / labels.shape[-1]
        )

    def _calibration_loss(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        losses = []

        for prediction, label in zip(predictions, labels):
            targets = label.nonzero()

            o = []
            t = []

            for target in targets:
                mask = label.clone().bool()
                mask[target] = 0
                mask = ~mask

                t.append(label[mask].nonzero()[0])
                o.append(prediction[mask].view(1, -1))

            if len(o) > 0 and len(t) > 0:
                o = torch.cat(o, dim=0)
                t = torch.cat(t, dim=0)

                losses.append(torch.nn.functional.cross_entropy(o, t, reduction="mean"))

        return torch.sum(torch.stack(losses, dim=0), dim=0)

    def loss(self, input: MMTInput, output: MMTOutput) -> torch.Tensor:
        nce_loss = self._nce_loss(output.projections.modalities)
        kd_loss = self._knowledge_distillation_loss(list(output.predictions.modalities.values()), output.predictions.fusion)
        classification_loss = self._classification_loss(
            output.predictions.fusion, input.labels
        )
        calibration_loss = self._calibration_loss(
            output.predictions.fusion, input.labels
        )

        return (
            classification_loss
            + self.config.calibration_weight * calibration_loss
            + self.config.nce_weight * nce_loss
            + self.config.kd_weight * kd_loss
        )

    def metric_representation(
        self, input: MMTInput, output: MMTOutput
    ) -> MMTMetricRepresentation:
        predictions = output.predictions.fusion.detach().sigmoid().round().cpu().numpy()
        labels = torch.where(input.labels == 0, 0, 1).cpu().numpy()

        return MultiModalTransformerModel.MMTMetricRepresentation(
            labels=labels, predictions=predictions
        )

    def collate_metrics(
        self, representations: typing.List[MMTMetricRepresentation]
    ) -> MMTMetricRepresentation:
        labels = []
        predictions = []

        for representation in representations:
            labels.extend(representation.labels.tolist())
            predictions.extend(representation.predictions.tolist())

        return MultiModalTransformerModel.MMTMetricRepresentation(
            np.array(labels), np.array(predictions)
        )

    def metrics(
        self, representation: MMTMetricRepresentation
    ) -> typing.Dict[str, float]:
        return {
            "f1-samples": round(
                sklearn.metrics.f1_score(
                    representation.labels,
                    representation.predictions,
                    average="samples",
                )
                * 100,
                2,
            ),
            "f1-weighted": round(
                sklearn.metrics.f1_score(
                    representation.labels,
                    representation.predictions,
                    average="weighted",
                )
                * 100,
                2,
            ),
            "classification report": sklearn.metrics.classification_report(
                representation.labels.round(),
                representation.predictions.round(),
                target_names=list(self.metadata[data.constants.META_CLASSES].keys()),
            ),
        }
