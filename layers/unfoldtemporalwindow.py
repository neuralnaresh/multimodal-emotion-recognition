import torch


class UnfoldTemporalWindow(torch.nn.Module):
    def __init__(
        self, window_size: int, window_stride: int, window_dilation: int = 1
    ) -> None:
        super().__init__()

        self.window_size = window_size

        self.unfold = torch.nn.Unfold(
            kernel_size=(window_size, 1),
            dilation=(window_dilation, 1),
            stride=(window_stride, 1),
            padding=(
                (window_size + (window_size - 1) * (window_dilation - 1) - 1) // 2,
                0,
            ),
        )

    def forward(self, x):
        """
        x: (batch_size, channels, frames, vertices)

        out: (batch_size, channels, frames, vertices * window_size)
        """

        batch_size = x.shape[0]
        channels = x.shape[1]
        vertices = x.shape[3]

        x = self.unfold(x)

        x = (
            x.view(batch_size, channels, self.window_size, -1, vertices)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        x = x.view(batch_size, channels, -1, self.window_size * vertices)

        return x
