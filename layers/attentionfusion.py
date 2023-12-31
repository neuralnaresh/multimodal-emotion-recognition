import enum

import torch

class AttentionFusionMechanism(enum.Enum):
    CONCAT = 'concat'
    HADAMARD = 'hadamard'

class AttentionFusion(torch.nn.Module):
    def __init__(self, modalities: int, input_size: int, pre_forward: bool = True, fusion: AttentionFusionMechanism = AttentionFusionMechanism.HADAMARD, norm: bool = True, dropout: float = 0.1) -> None:
        super().__init__()

        self.modalities = modalities
        self.pre_forward = pre_forward
        self.fusion = fusion
        self.norm = norm

        if pre_forward:
            self.pre_layers = torch.nn.ModuleList([
                torch.nn.Linear(input_size, input_size, bias=False)
                for _ in range(modalities)
            ])

        self.layernorm = torch.nn.LayerNorm(input_size, eps=1e-6)
        self.dropout = torch.nn.Dropout(dropout)

        if fusion == AttentionFusionMechanism.HADAMARD:
            self.fusion_layer = torch.nn.Linear(input_size, input_size, bias=False)
        elif fusion == AttentionFusionMechanism.CONCAT:
            self.fusion_layer = torch.nn.Linear(input_size, input_size, bias=False)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        assert len(inputs) == self.modalities, "Number of inputs must match number of modalities"

        for i in inputs:
            print(i.shape)

        if self.pre_forward:
            inputs = [self.pre_layers[i](input) for i, input in enumerate(inputs)]

        if self.fusion == AttentionFusionMechanism.HADAMARD:
            hidden = torch.prod(torch.stack(inputs), dim=0)
            if self.norm:
                hidden = self.layernorm(hidden)
        elif self.fusion == AttentionFusionMechanism.CONCAT:
            hidden = torch.cat(inputs, dim=1)

        hidden = self.dropout(hidden)
        return self.fusion_layer(hidden)