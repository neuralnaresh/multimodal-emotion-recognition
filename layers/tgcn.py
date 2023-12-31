import torch


class TGCN(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        temporal_kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dialation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size

        self.conv = torch.nn.Conv2d(
            input_channels,
            output_channels * kernel_size,
            kernel_size=(temporal_kernel_size, 1),
            padding=(padding, 0),
            stride=(stride, 1),
            dilation=(dialation, 1),
            bias=bias,
        )

    def forward(self, input, adjacency) -> torch.tensor:
        x = self.conv(input)

        n, kc, f, v = x.shape

        x = x.view(n, self.kernel_size, kc // self.kernel_size, f, v)
        x = torch.einsum("bkcfv,kvw->bcfw", (x, adjacency))

        return x
