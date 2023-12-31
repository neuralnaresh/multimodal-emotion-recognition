import torch

from layers.common import pad

from layers.stgcnencoder import STGCNEncoder

class DialogSTGCNEncoder(STGCNEncoder):
    def forward(self, input: torch.Tensor, utterance_lengths: torch.Tensor) -> torch.Tensor:
        '''
            input - (batch_size, utterances, frames, channels, vertices, persons)

            output - (batch_size, utterances, output_size)
        '''

        batch_size = input.shape[0]
        utterance_length = input.shape[1]

        outputs = []

        for i in range(batch_size):
            utterance_input = input[i, :utterance_lengths[i], :, :, :, :]
            outputs.append(super().forward(utterance_input))

        return pad(outputs, utterance_length)