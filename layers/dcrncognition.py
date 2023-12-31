import typing

import torch

from layers.common import pad_single

from layers.dcrnreason import DCRNReason

from layers.config.dcrncognition import DCRNCognitionConfig


class DCRNCognition(torch.nn.Module):
    def __init__(
        self, input_size: int, output_size: int, config: DCRNCognitionConfig
    ) -> None:
        super().__init__()

        self.fc = torch.nn.Linear(input_size, input_size * 2)
        self.steps = config.steps if config.steps is not None else [0, 0]
        self.reasons = torch.nn.ModuleList(
            [DCRNReason(input_size, step, layers=1) for step in self.steps]
        )
        self.dropout = torch.nn.Dropout(config.dropout)
        self.output = torch.nn.Linear(input_size * 4, output_size)

    def _feature_transfer(
        self,
        bank_s: torch.Tensor,
        bank_p: torch.Tensor,
        utterance_lengths: torch.Tensor,
    ):
        input_conversation_length = torch.tensor(utterance_lengths)
        max_conversation_length = max(utterance_lengths)

        start_zero = input_conversation_length.new(1).zero_()
        start = torch.cumsum(
            torch.cat((start_zero, input_conversation_length[:-1])), dim=0
        )

        bank_s = torch.stack(
            [
                pad_single(bank_s.narrow(0, s, l), max_conversation_length)
                for s, l in zip(
                    start.data.tolist(), input_conversation_length.data.tolist()
                )
            ],
            dim=0,
        ).transpose(0, 1)
        bank_p = torch.stack(
            [
                pad_single(bank_p.narrow(0, s, l), max_conversation_length)
                for s, l in zip(
                    start.data.tolist(), input_conversation_length.data.tolist()
                )
            ],
            dim=0,
        ).transpose(0, 1)

        return bank_s, bank_p

    def forward(self, input, speakers, utterance_lengths):
        batch_size = input.shape[1]

        batch_index = []

        context_s_ = []
        context_p_ = []

        for j in range(batch_size):
            batch_index.extend([j] * utterance_lengths[j])

            context_s_.append(input[: utterance_lengths[j], j, :])
            context_p_.append(speakers[: utterance_lengths[j], j, :])

        batch_index = torch.tensor(batch_index, device=input.device)

        bank_s_ = torch.cat(context_s_, dim=0)
        bank_p_ = torch.cat(context_p_, dim=0)

        bank_s_, bank_p_ = self._feature_transfer(bank_s_, bank_p_, utterance_lengths)

        features_s_ = []
        features_p_ = []

        for t in range(bank_s_.shape[0]):
            qstar = self.fc(bank_s_[t])
            qsitu = self.reasons[0](bank_s_, batch_index, qstar)
            features_s_.append(qsitu.unsqueeze(0))

        for t in range(bank_p_.shape[0]):
            qstar = self.fc(bank_p_[t])
            qspeaker = self.reasons[1](bank_p_, batch_index, qstar)
            features_p_.append(qspeaker.unsqueeze(0))

        features_s_ = torch.cat(features_s_, dim=0)
        features_p_ = torch.cat(features_p_, dim=0)

        hidden = self.dropout(torch.relu(torch.cat([features_s_, features_p_], dim=-1)))

        prob = torch.log_softmax(self.output(hidden), dim=-1)
        prob = torch.cat(
            [
                prob[:, j, :][: utterance_lengths[j]]
                for j in range(len(utterance_lengths))
            ]
        )

        return prob
