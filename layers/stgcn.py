import typing

import torch

from layers.tgcn import TGCN

class STGCN(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: typing.Tuple[int, int],
        stride: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__()

        self.gcn = TGCN(input_channels, output_channels, kernel_size[1])

        self.tcn = torch.nn.Sequential(
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(output_channels, output_channels, (kernel_size[0], 1), (stride, 1), ((kernel_size[0] - 1) // 2, 0)),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif input_channels == output_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = torch.nn.Sequential(
                torch.nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=(stride, 1)),
                torch.nn.BatchNorm2d(output_channels),
            )

        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, input, adjacenecy) -> torch.Tensor:
        res = self.residual(input)

        x = self.gcn(input, adjacenecy)
        x = self.tcn(x) + res

        return self.relu(x)