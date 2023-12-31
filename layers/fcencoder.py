import torch

from layers.config.fcencoder import FCEncoderConfig

class FCEncoder(torch.nn.Module):
    def __init__(self, input_size: int, config: FCEncoderConfig) -> None:
        super().__init__()

        self.dropout = torch.nn.Dropout(p=config.dropout)
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(input_size if i == 0 else config.hidden_size, config.hidden_size) for i in range(config.layers + 1)])
        self.output_layer = torch.nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x):
        x = self.dropout(x)

        for linear_layer in self.linear_layers:
            x = torch.tanh(linear_layer(x))

        return self.output_layer(x)
