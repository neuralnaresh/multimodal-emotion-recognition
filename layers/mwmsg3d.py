import torch

import numpy as np

from layers.msg3d import MSG3D


class MultiWindowMSG3D(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        adjacency_matrix: np.ndarray,
        scales: int,
        window_sizes=[3, 3],
        window_stride=1,
        window_dilations=[1, 1],
        initial_block: bool = False,
    ) -> None:
        super().__init__()

        self.g3ds = torch.nn.ModuleList(
            [
                MSG3D(
                    input_channels,
                    output_channels,
                    adjacency_matrix,
                    scales,
                    window_size,
                    window_stride,
                    window_dilation,
                    initial_block=initial_block
                )
                for window_size, window_dilation in zip(window_sizes, window_dilations)
            ]
        )

    def forward(self, x):
        out = 0

        for g3d in self.g3ds:
            out += g3d(x)
        
        return out
