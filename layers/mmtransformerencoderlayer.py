import typing

import torch

from layers.multiheadattention import MultiHeadAttention
from layers.attentionfusion import AttentionFusion
from layers.doublelinear import DoubleLinear

from layers.config.mmtransformerencoderlayer import (
    MultiModalTransformerEncoderLayerConfig,
)


class MultiModalTransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        modalities: int,
        config: MultiModalTransformerEncoderLayerConfig,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.modalities = modalities
        self.total = modalities + 1
        self.config = config

        self.layernorm = torch.nn.ModuleList(
            [torch.nn.LayerNorm(input_size, eps=config.epsilon) for _ in range(self.total)]
        )
        self.fusion = torch.nn.ModuleDict(
            {
                "key": AttentionFusion(
                    modalities,
                    input_size,
                    pre_forward=config.fusion.pre_forward,
                    fusion=config.fusion.mechanism,
                    norm=config.fusion.normalize,
                    dropout=config.dropout,
                ),
                "value": AttentionFusion(
                    modalities,
                    input_size,
                    pre_forward=config.fusion.pre_forward,
                    fusion=config.fusion.mechanism,
                    norm=config.fusion.normalize,
                    dropout=config.dropout,
                ),
            }
        )
        self.attention = torch.nn.ModuleList(
            [
                MultiHeadAttention(
                    input_size,
                    output_size,
                    key_value_size=config.key_value_size,
                    heads=config.heads,
                    bias=config.bias,
                    dropout=config.dropout,
                )
                for _ in range(self.total)
            ]
        )

        self.dropout = torch.nn.ModuleList(
            [torch.nn.Dropout(config.dropout) for _ in range(self.total)]
        )

        self.feed_forward = torch.nn.ModuleList(
            [
                DoubleLinear(
                    output_size,
                    config.feed_forward_size,
                    output_size,
                    activation=config.activation,
                    bias=config.bias,
                    dropout=config.dropout,
                )
                for _ in range(self.total)
            ]
        )

    def forward(
        self,
        inputs: list[torch.Tensor],
        attention_mask: list[typing.Optional[torch.Tensor]],
    ) -> list[dict[str, torch.Tensor]]:
        assert len(inputs) == self.total,             "Number of inputs must match number of modalities + fusion"
        assert len(attention_mask) == self.total,     "Number of attention masks must match number of modalities + fusion"

        residual = inputs

        if self.config.pre_normalize:
            inputs = [self.layernorm[i](input) for i, input in enumerate(inputs)]

        attention_outputs = [
            self.attention[i](
                query=input, key=input, value=input, mask=attention_mask[i]
            )
            for i, input in enumerate(inputs[: self.modalities])
        ] + [
            self.attention[-1](
                query=inputs[-1],
                key=self.fusion["key"](inputs[:-1]),
                value=self.fusion["value"](inputs[:-1]),
                mask=attention_mask[-1],
            )
        ]

        hidden = [attention_output["hidden"] for attention_output in attention_outputs]
        hidden = [self.dropout[i](hs) + residual[i] for i, hs in enumerate(hidden)]

        if not self.config.pre_normalize:
            hidden = [self.layernorm[i](hs) for i, hs in enumerate(hidden)]

        hidden = [self.feed_forward[i](hs) + hs for i, hs in enumerate(hidden)]
        
        for i, hs in enumerate(hidden):
            attention_outputs[i]["hidden"] = hs

        for h in hidden:
            print(h.shape)

        return attention_outputs
