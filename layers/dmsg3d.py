import torch

from layers.common import pad

from layers.mwmsg3dencoder import MultiWindowMSG3dEncoder

class DialogMultiWindowMSG3DEncoder(MultiWindowMSG3dEncoder):
    def forward(self, input: torch.Tensor, utterance_lengths: torch.Tensor):
        """
            x: (batch_size, utterances, channels, frames, vertices, persons)
            utterance_lengths: (batch_size)

            out: (batch_size, utterances, output_size)
        """

        batch_size = input.shape[0]
        utterance_length = input.shape[1]

        outputs = []

        for i in range(batch_size):
            utterance_input = input[i, :utterance_lengths[i], :, :, :, :]
            outputs.append(super().forward(utterance_input))

        return pad(outputs, utterance_length)
