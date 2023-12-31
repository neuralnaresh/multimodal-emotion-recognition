import typing
import functools

import torch

import sklearn.metrics

import numpy as np

from dlpipeline.model.model import Model
from dlpipeline.model.input import model_input, ModelInputConfig

import data.constants

from layers.fcencoder import FCEncoder
from layers.common import EncoderModuleBuild

from layers.fusion import Fusion
from layers.attentionbank import AttentionBank
from layers.graphormerencoder import GraphormerEncoder
from layers.dencoderbank import DialogEncoderBank

from losses.loss import Losses

from layers.config.fusion import FusionSource

from models.config.utterance import UtteranceConfig


class Utterance(Model):
    @model_input(
        {
            "indexes": ModelInputConfig(batch_first=True, dtype=torch.long),
            "text_embedding": ModelInputConfig(batch_first=True),
            "audio_features": ModelInputConfig(batch_first=True),
            "primary_face_aus": ModelInputConfig(batch_first=True),
            "primary_face_landmark_graph": ModelInputConfig(collate=GraphormerEncoder.collate),
            "frames": ModelInputConfig(batch_first=True),
            "audio": ModelInputConfig(batch_first=True),
            "audio_seq_length": ModelInputConfig(batch_first=True, dtype=torch.long),
            "face_features_seq_length": ModelInputConfig(
                batch_first=True, dtype=torch.long
            ),
            "labels": ModelInputConfig(batch_first=True),
        }
    )
    class UtteranceInput:
        indexes: torch.Tensor  # (batch_size)
        text_embedding: torch.Tensor # (batch_size, text_features)
        audio_features: torch.Tensor  # (batch_size, audio_features)
        primary_face_aus: torch.Tensor  # (batch_size, seq_length, au_count)
        primary_face_landmark_graph: typing.Dict[str, torch.Tensor]
        frames: torch.Tensor
        audio: torch.Tensor
        audio_seq_length: torch.Tensor
        face_features_seq_length: torch.Tensor  # (utterance_length, batch_size)
        labels: torch.Tensor  # (batch_size, num_labels)

    class UtteranceOutput(typing.NamedTuple):
        stages: typing.List[typing.Dict[str, torch.Tensor]]
        results: typing.List[typing.Union[torch.Tensor, None]]

    class UtteranceMetricRepresentation(typing.NamedTuple):
        labels: np.ndarray
        predictions: np.ndarray

    _INPUT_TYPE = UtteranceInput
    _OUTPUT_TYPE = UtteranceOutput
    _METRIC_REPRESENTATION = UtteranceMetricRepresentation

    def __init__(self, metadata: typing.Dict[str, typing.Any]) -> None:
        super().__init__()

        self.metadata = metadata

        config = UtteranceConfig()

        build = {
            data.constants.DATA_FRAMES: EncoderModuleBuild(),
            data.constants.DATA_AUDIO: EncoderModuleBuild(),
            data.constants.DATA_TEXT_EMBEDDING: EncoderModuleBuild(
                size=metadata[data.constants.META_TEXT_EMBEDDING_DIM]
            ),
            data.constants.DATA_AUDIO_FEATURES: EncoderModuleBuild(
                size=metadata[data.constants.META_AUDIO_FEATURE_DIM]
            ),
            data.constants.DATA_PRIMARY_FACE_AUS: EncoderModuleBuild(
                size=metadata[data.constants.META_FACE_AU_DIM]
            ),
            data.constants.DATA_PRIMARY_FACE_LANDMARK_GRAPH: EncoderModuleBuild(
                vertices=metadata[data.constants.META_FACE_LANDMARK_DIM],
                channels=metadata[data.constants.META_FACE_LANDMARK_COORD_DIM],
                persons=1,
                adjacency_matrix=np.array(
                    metadata[data.constants.META_FACE_LANDMARK_CONNECTIONS]
                ),
            ),
        }

        lengths = {
            data.constants.DATA_AUDIO: data.constants.DATA_AUDIO_SEQ_LENGTH,
            data.constants.DATA_FRAMES: data.constants.DATA_FACE_FEATURES_SEQ_LENGTH,
            data.constants.DATA_PRIMARY_FACE_HOG: data.constants.DATA_FACE_FEATURES_SEQ_LENGTH,
            data.constants.DATA_PRIMARY_FACE_AUS: data.constants.DATA_FACE_FEATURES_SEQ_LENGTH,
        }

        self.attention = AttentionBank(
            Utterance.UtteranceInput, config.attention, build=build, lengths=lengths
        )
        self.encoders = DialogEncoderBank(
            Utterance.UtteranceInput,
            config.encoders,
            self.attention._attention_meta,
            build=build,
            lengths=lengths,
        )
        self.fusions = torch.nn.ModuleList(
            [
                Fusion(fusion_config, self.encoders._encoder_meta)
                for fusion_config in config.fusions
            ]
        )
        self.losses = Losses(
            config.losses,
            [self.encoders._encoder_meta]
            + [fusion._fusion_meta for fusion in self.fusions],
            metadata,
        )

        classifer_config = UtteranceConfig.classifier
        classifer_config.output_size = metadata[data.constants.META_N_CLASSES]

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(functools.reduce(
                lambda size, meta: size + meta.build.size,
                self.fusions[-1]._fusion_meta.values() if len(self.fusions) > 0 else self.encoders._encoder_meta.values(),
                0,
            ), 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, metadata[data.constants.META_N_CLASSES])
        )

    def forward(self, input: UtteranceInput) -> UtteranceOutput:
        features = []

        attention = self.attention(input)
        encodings = self.encoders(input, attention)

        stages = [encodings]

        for i, fusion in enumerate(self.fusions):
            stages.append(
                fusion(
                    stages[-1]
                    if UtteranceConfig().fusions[i].source == FusionSource.FUSION
                    else stages[0]
                )
            )

        results = []
        classify = [c % len(stages) for c in UtteranceConfig().classify]

        for i, stage in enumerate(stages):
            if i in classify:
                features = torch.cat(list(stage.values()), dim=-1)

                results.append(torch.sigmoid(self.classifier(features)))
            else:
                results.append(None)

        return Utterance.UtteranceOutput(stages, results)

    def loss(self, input: UtteranceInput, output: UtteranceOutput) -> torch.Tensor:
        loss = self.losses(
            input,
            output.stages,
            output.results,
            input.labels
        )

        return loss

    def metric_representation(
        self, input: UtteranceInput, output: UtteranceOutput
    ) -> UtteranceMetricRepresentation:
        predictions = output.results[UtteranceConfig().classify[-1]].detach().cpu().numpy().round()
        labels = input.labels.cpu().numpy()

        return Utterance.UtteranceMetricRepresentation(labels, predictions)

    def collate_metrics(
        self, representations: typing.List[UtteranceMetricRepresentation]
    ) -> UtteranceMetricRepresentation:
        labels = []
        predictions = []

        for representation in representations:
            labels.extend(representation.labels.tolist())
            predictions.extend(representation.predictions.tolist())

        return Utterance.UtteranceMetricRepresentation(np.array(labels), np.array(predictions))

    def metrics(
        self, representation: UtteranceMetricRepresentation
    ) -> typing.Dict[str, float]:
        return {
            "accuracy": round(
                sklearn.metrics.accuracy_score(
                    representation.labels, representation.predictions
                )
                * 100,
                2,
            ),
            "f1": round(
                sklearn.metrics.f1_score(
                    representation.labels,
                    representation.predictions,
                    average="samples",
                )
                * 100,
                2,
            ),
            "classification report": sklearn.metrics.classification_report(
                representation.labels.round(),
                representation.predictions.round(), 
                target_names = list(self.metadata[data.constants.META_CLASSES].keys())
            )
        }
