import typing

import torch
import sklearn.metrics

from dlpipeline.model.model import Model
from dlpipeline.model.input import model_input

import data.constants

from layers.drnn import DRNN
from layers.fcencoder import FCEncoder
from layers.drnnencoder import DRNNEncoder
from layers.simplefusion import SimpleFusion
from layers.mwmsg3dencoder import MultiWindowMSG3dEncoder
from layers.drnnmatchingattention import DRNNMatchingAttention

from models.config.bidrnn import BiDRNNConfig


class BiDRNN(Model):
    @model_input
    class BiDRNNInput:
        text_embedding: torch.Tensor  #  (utterances, batch_size, embedding_dim)
        frame_embedding: torch.Tensor  # (utterances, batch_size, seq_len, embedding_dim)
        audio_embedding: torch.Tensor  # (utterances, batch_size, seq_len, embedding_dim)
        audio_features: torch.Tensor  # (utterances, batch_size, feature_dim)
        primary_face_hog: torch.Tensor  # (utterances, batch_size, seq_len, embedding_dim)
        primary_face_landmark_graph: torch.Tensor  # (utterances, batch_size, coordinates, seq_len, landmark_dim, 1)
        primary_face_aus: torch.Tensor  # (utterances, batch_size, seq_len, embedding_dim)
        frame_seq_lengths: torch.Tensor  # (utterances, batch_size)
        audio_seq_lengths: torch.Tensor  # (utterances, batch_size)
        face_features_seq_length: torch.Tensor  # (utterances, batch_size)
        speaker: torch.Tensor  # (utterances, batch_size, n_speakers)
        utterance_mask: torch.Tensor  # (utterances, batch_size)
        label: torch.Tensor  # (utterances, batch_size, n_classes)

    _INPUT_TYPE = BiDRNNInput

    def __init__(self, metadata: typing.Dict[str, typing.Any]) -> None:
        super().__init__()

        config = BiDRNNConfig()

        self._embedders = []
        self._fusion_size = 0

        self._create_encoders(metadata)

        self.fusion = SimpleFusion(
            self._fusion_size, metadata[data.constants.META_N_CLASSES], config.fusion
        )

        self.forward_drnn = DRNN(
            config.fusion_projection_hidden_dim,
            config.cell_global_state_dim,
            config.cell_participant_state_dim,
            config.cell_hidden_dim,
            config.cell_output_dim,
            config.cell_dropout,
        )

        self.reverse_drnn = DRNN(
            config.fusion_projection_hidden_dim,
            config.cell_global_state_dim,
            config.cell_participant_state_dim,
            config.cell_hidden_dim,
            config.cell_output_dim,
            config.cell_dropout,
        )

        self.dropout = torch.nn.Dropout(config.cell_dropout + 0.15)

        self.linear = torch.nn.Linear(
            2 * config.cell_output_dim, 2 * config.cell_hidden_dim
        )
        self.output = torch.nn.Linear(
            2 * config.cell_hidden_dim, metadata[data.constants.META_N_CLASSES]
        )

        if config.emotion_attention:
            self.emotion_attention = DRNNMatchingAttention(
                2 * config.cell_output_dim, 2 * config.cell_output_dim
            )

        self.criterion = torch.nn.NLLLoss()

    def _create_encoders(self, metadata: typing.Dict[str, typing.Any]) -> None:
        config = BiDRNNConfig()

        if config.encoders.text:
            self.text_encoder = FCEncoder(
                metadata[data.constants.META_TEXT_EMBEDDING_DIM], config.text
            )
            self._embedders.append(
                lambda input, utterances, batch_size, utterance_lengths: self.text_encoder(
                    input.text_embedding.view(-1, input.text_embedding.shape[-1])
                ).view(
                    utterances, batch_size, -1
                )
            )
            self._fusion_size += config.text.output_size

        if config.encoders.frames:
            self.frames_encoder = DRNNEncoder(
                metadata[data.constants.META_FRAME_EMBEDDIG_DIM], config.frames
            )
            self._embedders.append(
                lambda input, utterances, batch_size, utterance_lengths: self.frames_encoder(
                    input.frame_embedding, input.frame_seq_lengths, utterance_lengths
                )
            )
            self._fusion_size += config.frames.output_size

        if config.encoders.audio:
            self.audio_encoder = DRNNEncoder(
                metadata[data.constants.META_AUDIO_EMBEDDING_DIM], config.audio
            )
            self._embedders.append(
                lambda input, utterances, batch_size, utterance_lengths: self.audio_encoder(
                    input.audio_embedding, input.audio_seq_lengths, utterance_lengths
                )
            )
            self._fusion_size += config.audio.output_size

        if config.encoders.audio_features:
            self.audio_features_encoder = FCEncoder(
                metadata[data.constants.META_AUDIO_FEATURE_DIM], config.audio_features
            )
            self._embedders.append(
                lambda input, utterances, batch_size, utterance_lengths: self.audio_features_encoder(
                    input.audio_features.view(-1, input.audio_features.shape[-1])
                ).view(
                    utterances, batch_size, -1
                )
            )
            self._fusion_size += config.audio_features.output_size

        if config.encoders.primary_hog:
            self.primary_hog_encoder = DRNNEncoder(
                metadata[data.constants.META_FACE_HOG_DIM], config.primary_hog
            )
            self._embedders.append(
                lambda input, utterances, batch_size, utterance_lengths: self.primary_hog_encoder(
                    input.primary_face_hog,
                    input.face_features_seq_length,
                    utterance_lengths,
                )
            )
            self._fusion_size += config.primary_hog.output_size

        if config.encoders.secondary_hog:
            raise NotImplementedError

        if config.encoders.primary_landmarks:
            self.primary_landmarks_encoder = MultiWindowMSG3dEncoder(
                metadata[data.constants.META_FACE_LANDMARK_COORD_DIM],
                metadata[data.constants.META_FACE_LANDMARK_DIM],
                1,
                config.primary_landmarks,
                metadata[data.constants.META_FACE_LANDMARK_CONNECTIONS],
            )
            self._embedders.append(
                lambda input, utterances, batch_size, utterance_lengths: self.primary_landmarks_encoder(
                    input.primary_face_landmark_graph.contiguous().view(
                        utterances * batch_size,
                        metadata[data.constants.META_FACE_LANDMARK_COORD_DIM],
                        -1,
                        metadata[data.constants.META_FACE_LANDMARK_DIM],
                        1,
                    )
                ).view(
                    utterances, batch_size, -1
                )
            )

            self._fusion_size += config.primary_landmarks.output_size

        if config.encoders.secondary_landmarks:
            raise NotImplementedError

        if config.encoders.primary_aus:
            self.primary_aus_encoder = DRNNEncoder(
                metadata[data.constants.META_FACE_AU_DIM], config.primary_aus
            )
            self._embedders.append(
                lambda input, utterances, batch_size, utterance_lengths: self.primary_aus_encoder(
                    input.primary_face_aus,
                    input.face_features_seq_length,
                    utterance_lengths,
                )
            )
            self._fusion_size += config.primary_aus.output_size

        if config.encoders.secondary_aus:
            raise NotImplementedError

    def _reverse_sequence(self, input: torch.Tensor, mask: torch.Tensor):
        """
        input - (utterances, batch_size, *)
        mask - (batch_size, utterances)
        """

        _input = input.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(_input, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return torch.nn.utils.rnn.pad_sequence(xfs)

    def forward(
        self, input: BiDRNNInput
    ) -> typing.Tuple[torch.Tensor, typing.List[torch.Tensor]]:
        """
        output - (batch_size*utterances, n_classes)
        """
        utterances = input.text_embedding.shape[0]
        batch_size = input.text_embedding.shape[1]

        utterance_lengths = torch.sum(input.utterance_mask.squeeze(), dim=0)

        input.utterance_mask = input.utterance_mask.transpose(0, 1).squeeze()

        fusion_input = torch.cat(
            [
                embedder(input, utterances, batch_size, utterance_lengths)
                for embedder in self._embedders
            ],
            dim=-1,
        )

        fusion, _ = self.fusion(fusion_input.view(utterances * batch_size, -1))
        fusion = fusion.view(utterances, batch_size, -1)

        emotions_forward, alpha_forward = self.forward_drnn(fusion, input.speaker)
        emotions_forward = self.dropout(emotions_forward)

        reverse_fusion = self._reverse_sequence(fusion, input.utterance_mask)
        reverse_speakers = self._reverse_sequence(input.speaker, input.utterance_mask)

        emotions_reverse, alpha_reverse = self.reverse_drnn(
            reverse_fusion, reverse_speakers
        )
        emotions_reverse = self._reverse_sequence(
            emotions_reverse, input.utterance_mask
        )
        emotions_reverse = self.dropout(emotions_reverse)

        emotions = torch.cat((emotions_forward, emotions_reverse), dim=-1)

        if BiDRNNConfig().emotion_attention:
            emotion_attentions = []
            alphas = []

            for utterance_emotion in emotions:
                emotion_attention, alpha = self.emotion_attention(
                    emotions, utterance_emotion, input.utterance_mask
                )

                emotion_attentions.append(emotion_attention.unsqueeze(dim=0))
                alphas.append(alpha[:, 0, :])

            emotions = torch.cat(emotion_attentions, dim=0)

        hidden = torch.nn.functional.relu(self.linear(emotions))
        hidden = self.dropout(hidden)

        output = torch.nn.functional.log_softmax(self.output(hidden), dim=-1)
        return output.transpose(0, 1).contiguous().view(-1, output.shape[2])

    def loss(self, input: BiDRNNInput, prediction: torch.Tensor) -> torch.Tensor:
        return self.criterion(
            prediction * input.utterance_mask.contiguous().view(-1, 1),
            torch.argmax(input.label.transpose(0, 1), dim=-1).view(-1),
        ) / torch.sum(input.utterance_mask)

    def metrics(
        self, input: BiDRNNInput, prediction: torch.Tensor
    ) -> typing.Dict[str, float]:
        prediction = torch.argmax(prediction, dim=-1).cpu().numpy()
        labels = (
            torch.argmax(input.label.transpose(0, 1), dim=-1).view(-1).cpu().numpy()
        )
        mask = input.utterance_mask.contiguous().view(-1).cpu().numpy()

        return {
            "accuracy": round(
                sklearn.metrics.accuracy_score(labels, prediction, sample_weight=mask)
                * 100,
                2,
            ),
            "f1": round(
                sklearn.metrics.f1_score(
                    labels, prediction, sample_weight=mask, average="weighted"
                )
                * 100,
                2,
            ),
        }
