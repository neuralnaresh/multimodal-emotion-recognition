import torch


class TemporalConv(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        self.conv = torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=(kernel_size, 1),
            padding=((kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )
        self.batch_norm = torch.nn.BatchNorm2d(output_channels)

    def forward(self, x):
        return self.batch_norm(self.conv(x))
