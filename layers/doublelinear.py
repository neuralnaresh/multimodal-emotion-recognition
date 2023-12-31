import torch

from layers.config.mmtransformerencoderlayer import Activation

class DoubleLinear(torch.nn.Module):
    def __init__(self, input_size: int, internal_size: int, output_size: int, activation: Activation, bias: bool = False, norm = False, dropout: float = 0.1) -> None:
        super().__init__()

        self.normalize = norm

        self.input = torch.nn.Linear(input_size, internal_size, bias=bias)
        self.output = torch.nn.Linear(internal_size, output_size, bias=bias)

        self.dropout = torch.nn.Dropout(dropout)

        if norm:
            self.norm = torch.nn.BatchNorm1d(internal_size)

        if activation == Activation.RELU:
            self.activation = torch.nn.functional.relu
        elif activation == Activation.GELU:
            self.activation = torch.nn.functional.gelu
        elif activation == Activation.SWISH:
            self.activation = torch.nn.functional.silu
        elif activation == Activation.GEGLU:
            self.activation = torch.nn.functional.glu

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden = self.input(input)

        if self.normalize:
            hidden = self.norm(hidden)

        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.output(hidden)

        return hidden