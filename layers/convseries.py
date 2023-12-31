import typing

import torch

class ConvSeries(torch.nn.Module):
    def __init__(self, input_size: int, output_sizes: typing.List[int], dropout: float) -> None:
        super().__init__()

        layer_sizes = [input_size] + output_sizes

        self.layers = torch.nn.ModuleList()

        for i in range(1, len(layer_sizes)):
            self.layers.append(torch.nn.Dropout(p=dropout))
            self.layers.append(torch.nn.Conv2d(layer_sizes[i - 1], layer_sizes[i], kernel_size=1))
            self.layers.append(torch.nn.BatchNorm2d(layer_sizes[i]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x