import typing

import torch

from layers.temporalconv import TemporalConv


class MSTCN(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilations: typing.List[int] = [1, 2, 3, 4],
        residual: bool = True,
        residual_kernel_size: int = 1,
    ) -> None:
        super().__init__()

        self.num_branches = len(dilations) + 2
        branch_channels = output_channels // self.num_branches

        self.branches = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    input_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                torch.nn.BatchNorm2d(branch_channels),
                torch.nn.ReLU(),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation),
            )
            for dilation in dilations
        ])

        self.branches.append(torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, branch_channels, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(branch_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            torch.nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            torch.nn.BatchNorm2d(branch_channels)
        ))

        if not residual:
            self.residual = lambda x: 0
        elif (input_channels == output_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(input_channels, output_channels, kernel_size=residual_kernel_size, stride=stride)

    def forward(self, x):
        residual = self.residual(x)

        branch_outs = [branch(x) for branch in self.branches]

        out = torch.cat(branch_outs, dim=1) + residual
        out = torch.nn.functional.relu(out, inplace=True)

        return out
