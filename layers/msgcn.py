import torch

import numpy as np

import data.utils

from layers.convseries import ConvSeries

class MSGCN(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        scales: int,
        adjacency_matrix: np.ndarray,
        dropout: float = 0,
        disentangle: bool = True,
        adjacency_residual: bool = True,
    ) -> None:
        super().__init__()

        self.scales = scales

        if disentangle:
            adjacency_scales = [data.utils.k_adjacency(adjacency_matrix, k, with_self=True) for k in range(scales)]
            adjacency_scales = np.concatenate([data.utils.normalize_adjacency(a) for a in adjacency_scales])
        else:
            adjacency_scales = [data.utils.normalize_adjacency(adjacency_matrix)] * scales
            adjacency_scales = [np.linalg.matrix_power(a, k) for k, a in enumerate(adjacency_scales)]
            adjacency_scales = np.concatenate(adjacency_scales)

        self.adjacency_scales = torch.tensor(adjacency_scales)

        if adjacency_residual:
            self.adjacency_residual = torch.nn.init.uniform_(torch.nn.Parameter(torch.Tensor(self.adjacency_scales.shape)), -1e-6, 1e-6)
        else:
            self.adjacency_residual = torch.tensor(0)

        self.mlp = ConvSeries(input_channels * scales, [output_channels], dropout=dropout)

    def forward(self, x):
        """
            x: (batch_size, channels, windows, vertices)
        """

        batch_size, channels, windows, vertices = x.shape

        adjacency = self.adjacency_scales.to(x.dtype).to(x.device) + self.adjacency_residual.to(x.dtype).to(x.device)

        aggregation = torch.einsum("vu,bcwu->bcwv", adjacency, x)
        aggregation = aggregation.view(batch_size, channels, windows, self.scales, vertices)
        aggregation = aggregation.permute(0, 3, 1, 2, 4).contiguous().view(batch_size, self.scales*channels, windows, vertices)

        out = self.mlp(aggregation)
        
        return out
