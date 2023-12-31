import torch

from layers.drnnsimpleattention import DRNNSimpleAttention


class DRNNCell(torch.nn.Module):
    def __init__(
        self,
        feature_size: int,
        global_state_size: int,
        party_state_size: int,
        output_size: int,
        use_listener_state: bool = False,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.feature_size = feature_size
        self.global_state_size = global_state_size
        self.party_state_size = party_state_size
        self.output_size = output_size
        self.use_listener_state = use_listener_state

        self.global_state_cell = torch.nn.GRUCell(
            feature_size + party_state_size, global_state_size
        )
        self.party_state_cell = torch.nn.GRUCell(
            feature_size + global_state_size, party_state_size
        )
        self.output_cell = torch.nn.GRUCell(party_state_size, output_size)

        if use_listener_state:
            self.listener_state_cell = torch.nn.GRUCell(
                feature_size + party_state_size, party_state_size
            )

        self.dropout = torch.nn.Dropout(p=dropout)
        self.attention = DRNNSimpleAttention(global_state_size)

    def _select_parties(self, x, indices):
        party_selections = []

        for i, j in zip(indices, x):
            party_selections.append(j[i].unsqueeze(0))

        return torch.cat(party_selections, dim=0)

    def forward(self, input: torch.Tensor, party_mask: torch.Tensor, global_history: torch.Tensor, initial_party: torch.Tensor, initial_output: torch.Tensor):
        """
            input: (batch_size, input_size)
            party_mask: (batch_size, n_speakers)
            global_history: (t-1, batch_size, global_state_size)
            initial_party: (batch_size, n_speakers, party_state_size)
            initial_output: (batch_size, output_size)
        """

        batch_size = input.shape[0]

        party_mask_index = torch.argmax(party_mask, dim=1)
        party_selections = self._select_parties(initial_party, party_mask_index)

        global_state = self.global_state_cell(
            torch.cat([input, party_selections], dim=1),
            torch.zeros(batch_size, self.global_state_size).type(input.type())
            if global_history.shape[0] == 0
            else global_history[-1],
        )
        global_state = self.dropout(global_state)

        if global_history.shape[0] == 0:
            context = torch.zeros(batch_size, self.global_state_size).type(input.type())
            alpha = None
        else:
            context, alpha  = self.attention(global_history)

        input_context = torch.cat([input, context], dim=1).unsqueeze(1).expand(-1, party_mask.shape[1], -1)

        party_state = self.party_state_cell(
            input_context.contiguous().view(-1, self.feature_size + self.global_state_size), 
            initial_party.view(-1, self.party_state_size)
        ).view(batch_size, -1, self.party_state_size)
        party_state = self.dropout(party_state)

        if self.use_listener_state:
            raise NotImplementedError
        else:
            listener_state = initial_party

        party_mask_new = party_mask.unsqueeze(2)
        party_new = listener_state * (1 - party_mask_new) + party_state * party_mask_new
        
        if initial_output.shape[0] == 0:
            initial_output = torch.zeros(batch_size, self.output_size).type(input.type())
        else:
            initial_output = initial_output

        output = self.output_cell(self._select_parties(party_new, party_mask_index), initial_output)
        output = self.dropout(output)

        return global_state, party_new, output, alpha