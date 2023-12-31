import torch
import torch.nn.utils.rnn

from layers.common import pad

from layers.config.drnnencoder import DRNNEncoderConfig

class DRNNEncoder(torch.nn.Module):
    def __init__(self, input_size: int, config: DRNNEncoderConfig) -> None:
        super().__init__()

        self.rnn = torch.nn.LSTM(input_size, config.hidden_size, config.layers, dropout=config.dropout, bidirectional=config.bidirectional, batch_first=True)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.linear = torch.nn.Linear((2 if config.bidirectional else 1) * config.hidden_size, config.output_size)

        self.bidirectional = config.bidirectional

    def forward(self, x, lengths, utterance_lengths):
        '''
            x: (batch_size, utterances, seq_length, input_size)
            lengths: (batch_size, utterances)
            utterance_length: (batch_size)

            output: (batch_size, utterances, output_size)
        '''

        batch_size = x.shape[0]
        utterance_length = x.shape[1]

        outputs = []

        for b in range(batch_size):
            utterance_input = x[b, :utterance_lengths[b], :, :]
            seq_length = lengths[b, :utterance_lengths[b]]

            packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(utterance_input, seq_length.int().cpu(), batch_first=True, enforce_sorted=False)
            _, final_states = self.rnn(packed_sequence)

            if self.bidirectional:
                hidden = self.dropout(torch.cat((final_states[0][0], final_states[0][1]), dim=1))
            else:
                hidden = self.dropout(final_states[0].squeeze())

            output = self.linear(hidden)
            outputs.append(output)

        return pad(outputs, utterance_length)
