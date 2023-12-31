import torch

from layers.config.simplefusion import FusionConfig

class SimpleFusion(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, config: FusionConfig) -> None:
        super(SimpleFusion, self).__init__()

        self.batch_norm = torch.nn.BatchNorm1d(input_size)
        self.dropout = torch.nn.Dropout(p=config.dropout)
        
        self.linear_1 = torch.nn.Linear(input_size, config.hidden_size)
        self.linear_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_3 = torch.nn.Linear(config.hidden_size, output_size)

    def forward(self, x):
        '''
            x: (batch_size, input_size, *)

            out_2: (batch_size, hidden_size)
            out_3: (batch_size, classes, *)
        '''

        x = self.batch_norm(x)
        x = self.dropout(x)

        out_1 = torch.nn.functional.tanh(self.linear_1(x))
        out_2 = torch.nn.functional.tanh(self.linear_2(out_1))
        out_3 = self.linear_3(out_2)

        return out_2, out_3