import torch

import numpy as np

from layers.convseries import ConvSeries
from layers.spatialtemporalmsgcn import SpatialTemporalMSGCN
from layers.unfoldtemporalwindow import UnfoldTemporalWindow


class MSG3D(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        adjacency_matrix: np.ndarray,
        scales: int,
        window_size: int,
        window_stride: int,
        window_dilation: int,
        embed_factor: int = 1,
        initial_block: bool = False
    ) -> None:
        super().__init__()

        self.window_size = window_size

        self.embed_channels_in = output_channels // embed_factor
        self.embed_channels_out = output_channels // embed_factor

        if embed_factor == 1:
            self.input = torch.nn.Identity()

            self.embed_channels_in = input_channels
            self.embed_channels_out = input_channels

            if initial_block:
                self.embed_channels_out = output_channels
        else:
            self.input = ConvSeries(input_channels, [self.embed_channels_in])

        self.utw = UnfoldTemporalWindow(window_size, window_stride, window_dilation)
        self.stgcn = SpatialTemporalMSGCN(
                self.embed_channels_in,
                self.embed_channels_out,
                adjacency_matrix,
                scales,
                window_size,
                adjacency_residual=True,
        )

        self.conv_out = torch.nn.Conv3d(
            self.embed_channels_out, output_channels, kernel_size=(1, window_size, 1)
        )
        self.batch_norm_out = torch.nn.BatchNorm2d(output_channels)

    def forward(self, x):
        """
            x: (batch_size, channels, frames, vertices)
        """

        batch_size = x.shape[0]
        vertices = x.shape[3]

        x = self.input(x)
        x = self.utw(x)
        x = self.stgcn(x)

        x = x.view(batch_size, self.embed_channels_out, -1, self.window_size, vertices)

        x = self.conv_out(x).squeeze(dim=3)
        x = self.batch_norm_out(x)

        return x