from locale import normalize
import torch

import numpy as np

from layers.convseries import ConvSeries
from layers.msgcn import MSGCN


class SpatialTemporalMSGCN(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        adjacency_matrix: np.ndarray,
        scales: int,
        window_size: int,
        dropout: float = 0.2,
        disentangle: bool = True,
        adjacency_residual: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()

        self.msgcn = MSGCN(
            input_channels,
            output_channels,
            scales,
            self._build_spatial_temporal_graph(adjacency_matrix, window_size),
            dropout,
            disentangle,
            adjacency_residual,
        )

        if not residual:
            self.residual = lambda x: 0
        elif input_channels == output_channels:
            self.residual = lambda x: x
        else:
            self.residual = ConvSeries(
                input_channels, [output_channels], dropout=dropout
            )

    def _build_spatial_temporal_graph(
        self, adjacency_matrix: np.ndarray, window_size: int
    ) -> np.ndarray:
        vertices = adjacency_matrix.shape[0]

        adjacency_matrix_identity = adjacency_matrix + np.eye(
            vertices, dtype=adjacency_matrix.dtype
        )
        adjacency_matrix_identity_large = np.tile(
            adjacency_matrix_identity, (window_size, window_size)
        )

        return adjacency_matrix_identity_large

    def forward(self, x):
        """
        x: (batch_size, channels, windows, vertices)
        """

        residual = self.residual(x)

        out = self.msgcn(x)
        out += residual

        return torch.nn.functional.relu(out, inplace=True)
