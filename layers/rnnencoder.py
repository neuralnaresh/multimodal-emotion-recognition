import torch
import torch.nn.utils.rnn

from layers.config.rnnencoder import RNNEncoderConfig

class RNNEncoder(torch.nn.Module):
    def __init__(self, input_size: int, config: RNNEncoderConfig) -> None:
        super().__init__()

        self.rnn = torch.nn.LSTM(input_size, config.hidden_size, config.layers, dropout=config.dropout, bidirectional=config.bidirectional, batch_first=True)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.linear = torch.nn.Linear((2 if config.bidirectional else 1) * config.hidden_size, config.output_size)

        self.bidirectional = config.bidirectional

    def forward(self, x, lengths):
        lengths = lengths.int().cpu()

        packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False, batch_first=True)
        _, final_states = self.rnn(packed_sequence)

        if self.bidirectional:
            hidden = self.dropout(torch.cat((final_states[0][0], final_states[0][1]), dim=1))
        else:
            hidden = self.dropout(final_states[0].squeeze())
        
        output = self.linear(hidden)

        return output