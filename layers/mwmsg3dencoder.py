import typing

import torch

import numpy as np

from layers.mwmsg3d import MultiWindowMSG3D
from layers.msgcn import MSGCN
from layers.mstcn import MSTCN

from layers.config.mwmsg3dencoder import MultiWindowMSG3DEncoderConfig

class MultiWindowMSG3dEncoder(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        input_vertices: int,
        input_persons: int,
        config: MultiWindowMSG3DEncoderConfig,
        adjacency_matrix: np.ndarray,
        channels: typing.List[int] = [96, 192, 384],
    ) -> None:
        super().__init__()

        channels = [input_channels] + channels

        self.data_batch_norm = torch.nn.BatchNorm1d(input_channels * input_vertices * input_persons)
        self.stgc_blocks = torch.nn.ModuleList()
        self.fc = torch.nn.Linear(channels[-1], config.output_size)

        for i in range(len(channels) - 1):
            initial = i == 0

            self.stgc_blocks.append(
                torch.nn.ModuleList([
                    MultiWindowMSG3D(
                        channels[i],
                        channels[i + 1],
                        adjacency_matrix,
                        config.g3d_scales,
                        window_stride=(1 if initial else 2),
                        initial_block=initial,
                    ),
                    torch.nn.Sequential(
                        MSGCN(
                            channels[i],
                            channels[i + 1 if initial else i],
                            config.gcn_scales,
                            adjacency_matrix,
                            disentangle=True,
                        ),
                        MSTCN(
                            channels[i + 1 if initial else i], channels[i + 1], stride=(1 if initial else 2)
                        ),
                        MSTCN(channels[i + 1], channels[i + 1]),
                    ),
                    MSTCN(channels[i + 1], channels[i + 1]),
                ])
            )

    def forward(self, x):
        """
            x: (batch_size, channels, frames, vertices, persons)

            out: (batch_size, output_size)
        """
        batch_size, channels, frames, vertices, persons = x.shape

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(batch_size, persons * vertices * channels, frames)
        x = self.data_batch_norm(x)
        x = x.view(batch_size * persons, vertices, channels, frames).permute(0, 2, 3, 1).contiguous()

        for stgc in self.stgc_blocks:
            gcn3d, sgcn, tcn = stgc

            x = torch.nn.functional.relu(sgcn(x) + gcn3d(x), inplace=True)
            x = tcn(x)

        out = self.fc(x.view(batch_size, persons, x.shape[1], -1).mean(dim=3).mean(dim=1))

        return out
