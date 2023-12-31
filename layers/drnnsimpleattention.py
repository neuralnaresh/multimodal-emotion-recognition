import torch

class DRNNSimpleAttention(torch.nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()

        self.scalar = torch.nn.Linear(input_size, 1, bias=False)

    def forward(self, x):
        '''
            x: (utterances, batch_size, input_size)

            alpha: (batch_size, 1, utterances)
            attention: (batch_size, input_size)
        '''

        scalar = self.scalar(x)
        alpha = torch.nn.functional.softmax(scalar, dim=0).permute(1, 2, 0)
        attention = torch.bmm(alpha, x.transpose(0, 1))[:, 0, :]

        return attention, alpha