import torch

import numpy as np

from layers.stgcn import STGCN

from layers.config.stgcnencoder import STGCNEncoderConfig


class STGCNEncoder(torch.nn.Module):
    def __init__(
        self,
        vertices: int,
        channels: int,
        adjacency_matrix: np.ndarray,
        config: STGCNEncoderConfig,
    ) -> None:
        super().__init__()

        self.adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        if self.adjacency_matrix.dim() == 2:
            self.adjacency_matrix = self.adjacency_matrix.unsqueeze(dim=0)

        dimensions = [channels] + config.channels

        self.data_norm = torch.nn.BatchNorm1d(channels * vertices)
        self.gcns = torch.nn.ModuleList(
            [
                STGCN(
                    dimensions[i],
                    dimensions[i + 1],
                    (config.temporal_kernel_size, vertices),
                    dropout=config.dropout,
                    stride=(dimensions[i + 1] // dimensions[i] if i > 0 else 1),
                )
                for i in range(len(dimensions) - 1)
            ]
        )

        if config.importance_weighting:
            self.importance = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.ones(vertices, dtype=torch.float32)) for _ in dimensions]
            )
        else:
            self.importance = [1] * len(dimensions)

        self.fc = torch.nn.Conv2d(dimensions[-1], config.output_size, kernel_size=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (batch_size, frames, channels, vertices, persons)

        output: (batch_size, output_size)
        """

        B, F, C, V, P = input.shape

        x = input.permute(0, 4, 3, 2, 1).contiguous().view(B * P, V * C, F)
        x = self.data_norm(x)

        x = (
            x.view(B, P, V, C, F)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(B * P, C, F, V)
        )

        for gcn, importance in zip(self.gcns, self.importance):
            x = gcn(x, self.adjacency_matrix.to(x.device) * importance.to(x.device))

        x = torch.nn.functional.avg_pool2d(x, (x.shape[2:]))
        x = x.view(B, P, -1, 1, 1).mean(dim=1)

        x = self.fc(x)
        x = x.view(B, -1)

        return x
