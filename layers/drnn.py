import typing

import torch

from layers.drnncell import DRNNCell


class DRNN(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        global_state_size: int,
        participant_state_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.participant_state_size = participant_state_size

        self.dialog_cell = DRNNCell(
            input_size,
            global_state_size,
            participant_state_size,
            output_size,
            dropout=dropout,
        )

    def forward(
        self, features: torch.Tensor, speakers: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, typing.List[torch.Tensor]]:
        """
            features: (utterances, batch_size, input_size)
            speakers: (utterances, batch_size, n_speakers)

            output: (utterances, batch_size, output_size)
        """

        global_history = torch.zeros(0).type(features.type())
        parties = torch.zeros(speakers.shape[1], speakers.shape[2], self.participant_state_size).type(
            features.type()
        )
        new_output = torch.zeros(0).type(features.type())
        output = new_output

        alpha = []

        for update, party_mask in zip(features, speakers):
            global_state, parties, new_output, new_alpha = self.dialog_cell(
                update, party_mask, global_history, parties, new_output
            )
            global_history = torch.cat(
                (global_history, global_state.unsqueeze(0)), dim=0
            )
            output = torch.cat((output, new_output.unsqueeze(0)), dim=0)

            if new_alpha is not None:
                alpha.append(new_alpha[:, 0, :])

        return output, alpha
