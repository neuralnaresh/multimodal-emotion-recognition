import typing
import functools

import torch

import sklearn.metrics

import numpy as np

from dlpipeline.model.model import Model
from dlpipeline.model.input import model_input, ModelInputConfig

import data.constants

from layers.common import EncoderModuleBuild

from layers.fusion import Fusion
from layers.drnnencoder import DRNNEncoder
from layers.dcrncognition import DCRNCognition
from layers.attentionbank import AttentionBank
from layers.dencoderbank import DialogEncoderBank

from losses.loss import Losses

from layers.config.fusion import FusionSource
from layers.config.drnnencoder import DRNNEncoderConfig

from models.config.dcrn import DCRNBase, DCRNConfig


class DCRN(Model):
    @model_input(
        {
            "indexes": ModelInputConfig(batch_first=True, dtype=torch.long),
            "text_embedding": ModelInputConfig(batch_first=True),
            "audio_features": ModelInputConfig(batch_first=True),
            "primary_face_hog": ModelInputConfig(batch_first=True),
            "primary_face_aus": ModelInputConfig(batch_first=True),
            "primary_face_landmark_graph": ModelInputConfig(batch_first=True),
            "face_features_seq_length": ModelInputConfig(
                batch_first=True, dtype=torch.long
            ),
            "speaker_mask": ModelInputConfig(batch_first=True),
            "utterance_mask": ModelInputConfig(batch_first=True),
            "labels": ModelInputConfig(batch_first=True),
        }
    )
    class DCRNInput:
        indexes: torch.Tensor  # (batch_size, utterance_length)
        text_embedding: torch.Tensor  # (batch_size, utterance_length, embedding_size)
        audio_features: torch.Tensor  # (batch_size, utterance_length, audio_features)
        primary_face_hog: torch.Tensor  # (utterance_length, batch_size, seq_length, hog_feature_size)
        primary_face_aus: torch.Tensor  # (utterance_length, batch_size, seq_length, au_count)
        primary_face_landmark_graph: torch.Tensor  # (utterance_length, batch_size, ...)
        primary_face_aus_embedding: torch.Tensor # (utterance_length, batch_size, embedding_size)
        face_features_seq_length: torch.Tensor  # (utterance_length, batch_size)
        speaker_mask: torch.Tensor  # (batch_size, utterance_length, speaker_count)
        utterance_mask: torch.Tensor  # (batch_size, utterance_length)
        labels: torch.Tensor  # (batch_size, utterance_length, num_labels)

    class DCRNOutput(typing.NamedTuple):
        stages: typing.List[typing.Dict[str, torch.Tensor]]
        results: typing.List[typing.Union[torch.Tensor, None]]

    class DCRNMetricRepresentation(typing.NamedTuple):
        labels: np.ndarray
        predictions: np.ndarray

    _INPUT_TYPE = DCRNInput
    _OUTPUT_TYPE = DCRNOutput
    _METRIC_REPRESENTATION = DCRNMetricRepresentation

    def __init__(self, metadata: typing.Dict[str, typing.Any]) -> None:
        super().__init__()

        config = DCRNConfig()

        build = {
            data.constants.DATA_TEXT_EMBEDDING: EncoderModuleBuild(
                size=metadata[data.constants.META_TEXT_EMBEDDING_DIM]
            ),
            data.constants.DATA_AUDIO_FEATURES: EncoderModuleBuild(
                size=metadata[data.constants.META_AUDIO_FEATURE_DIM]
            ),
            data.constants.DATA_PRIMARY_FACE_HOG: EncoderModuleBuild(
                size=metadata[data.constants.META_FACE_HOG_DIM]
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
            data.constants.DATA_PRIMARY_FACE_AUS_EMBEDDING:  EncoderModuleBuild(size=128)
        }

        lengths = {
            data.constants.DATA_PRIMARY_FACE_HOG: data.constants.DATA_FACE_FEATURES_SEQ_LENGTH,
            data.constants.DATA_PRIMARY_FACE_AUS: data.constants.DATA_FACE_FEATURES_SEQ_LENGTH,
        }

        self.au_embeding = DRNNEncoder(metadata[data.constants.META_FACE_AU_DIM], DRNNEncoderConfig(output_size=128))

        self.attention = AttentionBank(
            DCRN.DCRNInput, config.attention, build=build, lengths=lengths
        )
        self.encoders = DialogEncoderBank(
            DCRN.DCRNInput,
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

        if config.base == DCRNBase.LSTM:
            self.utterance_rnn = torch.nn.LSTM(
                input_size=functools.reduce(
                    lambda size, meta: size + meta.build.size,
                    self.fusions[-1]._fusion_meta.values(),
                    0,
                ),
                hidden_size=config.hidden,
                num_layers=config.layers,
                dropout=config.dropout,
                bidirectional=True,
            )

            self.speaker_rnn = torch.nn.LSTM(
                input_size=functools.reduce(
                    lambda size, meta: size + meta.build.size,
                    self.fusions[-1]._fusion_meta.values(),
                    0,
                ),
                hidden_size=config.hidden,
                num_layers=config.layers,
                dropout=config.dropout,
                bidirectional=True,
            )

        else:
            raise NotImplementedError

        self.cognition = DCRNCognition(
            2 * config.hidden, metadata[data.constants.META_N_CLASSES], config.cognition
        )

    def _utterance_lengths(self, utterance_mask: torch.Tensor) -> typing.List[int]:
        return [
            (utterance_mask[j] == 1).nonzero().tolist()[-1][0] + 1
            for j in range(len(utterance_mask.data.tolist()))
        ]

    def forward(self, input: DCRNInput) -> DCRNOutput:
        features = []

        utterance_lengths = self._utterance_lengths(input.utterance_mask)

        input.primary_face_aus_embedding = self.au_embeding(input.primary_face_aus, input.face_features_seq_length, utterance_lengths)

        attention = self.attention(input, utterance_lengths)
        encodings = self.encoders(input, attention, utterance_lengths)

        stages = [encodings]

        for i, fusion in enumerate(self.fusions):
            stages.append(
                fusion(
                    stages[-1]
                    if DCRNConfig().fusions[i].source == FusionSource.FUSION
                    else stages[0]
                )
            )

        results = []
        classify = [c % len(stages) for c in DCRNConfig().classify]

        for i, stage in enumerate(stages):
            if i in classify:
                features = torch.cat(list(stage.values()), dim=-1)

                self.utterance_rnn.flatten_parameters()
                self.speaker_rnn.flatten_parameters()

                if DCRNConfig().base == DCRNBase.LSTM:
                    utterances = torch.zeros(
                        features.shape[0], features.shape[1], 200
                    ).type(features.type())
                    speakers = [
                        torch.zeros_like(features).type(features.type())
                        for _ in range(input.speaker_mask.shape[-1])
                    ]

                    for b in range(features.shape[0]):
                        for p in range(len(speakers)):
                            index = torch.nonzero(input.speaker_mask[b][:, p]).squeeze(
                                -1
                            )
                            if index.shape[0] == 0:
                                speakers[p][b][: index.shape[0]] = features[b][index]

                    speaker_emotions = [
                        self.speaker_rnn(speakers[p].transpose(0, 1))[0].transpose(0, 1)
                        for p in range(len(speakers))
                    ]

                    for b in range(utterances.shape[0]):
                        for p in range(len(speakers)):
                            index = torch.nonzero(input.speaker_mask[b][:, p]).squeeze(
                                -1
                            )
                            if index.shape[0] > 0:
                                utterances[b][index] = speaker_emotions[p][b][
                                    : index.shape[0]
                                ]

                    u_p = utterances.transpose(0, 1)
                    u_s, _ = self.utterance_rnn(features.transpose(0, 1))

                results.append(
                    self.cognition(
                        u_s, u_p, self._utterance_lengths(input.utterance_mask)
                    )
                )
            else:
                results.append(None)

        return DCRN.DCRNOutput(stages, results)

    def loss(self, input: DCRNInput, output: DCRNOutput) -> torch.Tensor:
        loss = self.losses(
            input,
            output.stages,
            output.results,
            input.labels,
            self._utterance_lengths(input.utterance_mask),
        )

        return loss

    def metric_representation(self, input: DCRNInput, output: DCRNOutput) -> DCRNMetricRepresentation:
        utterance_lengths = self._utterance_lengths(input.utterance_mask)

        predictions = output.results[DCRNConfig().classify[-1]]
        predictions = torch.argmax(predictions, dim=1).cpu().numpy()

        labels = torch.argmax(input.labels, dim=-1)
        labels = (
            torch.cat(
                [
                    labels[j][: utterance_lengths[j]]
                    for j in range(labels.shape[0])
                ]
            )
            .cpu()
            .numpy()
        )

        return DCRN.DCRNMetricRepresentation(labels, predictions)

    def collate_metrics(self, representations: typing.List[DCRNMetricRepresentation]) -> DCRNMetricRepresentation:
        labels = []
        predictions = []

        for representation in representations:
            labels.extend(representation.labels.tolist())
            predictions.extend(representation.predictions.tolist())

        return DCRN.DCRNMetricRepresentation(
            np.array(labels), np.array(predictions)
        )

    def metrics(self, representation: DCRNMetricRepresentation) -> typing.Dict[str, float]:
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
                    average="weighted",
                )
                * 100,
                2,
            ),
        }
