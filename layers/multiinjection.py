import math
import typing

import torch

from layers.fcencoder import FCEncoder

from layers.config.fcencoder import FCEncoderConfig
from layers.config.multiinjection import MultiInjectionConfig


class MultiInjection(torch.nn.Module):
    def __init__(
        self, input_size: int, payload_sizes: typing.List[int], config: MultiInjectionConfig
    ) -> None:
        super().__init__()

        for payload_size in payload_sizes:
            if payload_size % config.heads != 0:
                raise ValueError(
                    "Injection payload size must be divisible by the number of attention heads"
                )

        self.heads = config.heads
        self.query_size = config.payload_output_size // config.heads
        self.scale = math.sqrt(float(self.query_size))

        self.query_weight = torch.nn.parameter.Parameter(
            torch.Tensor(self.heads, config.hidden_size, self.query_size)
        )
        self.key_weights = torch.nn.ParameterList(
            [
                torch.Tensor(self.heads, payload_size, payload_size // config.heads)
                for payload_size in payload_sizes
            ]
        )
        self.value_weights = torch.nn.ParameterList(
            [
                torch.Tensor(self.heads, payload_size, payload_size // config.heads)
                for payload_size in payload_sizes
            ]
        )
        self.attention_weight = torch.nn.parameter.Parameter(
            torch.Tensor(self.heads * self.query_size, config.hidden_size)
        )

        self.dropout = torch.nn.Dropout(p=config.dropout)
        self.linear_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    input_size if i == 0 else config.hidden_size, config.hidden_size
                )
                for i in range(config.layers + 1)
            ]
        )
        self.output_layer = torch.nn.Linear(config.hidden_size, config.output_size)

        self.key_networks = torch.nn.ModuleList(
            [
                FCEncoder(
                    payload_size // config.heads,
                    FCEncoderConfig(
                        hidden_size=config.payload_key_size,
                        output_size=config.payload_output_size,
                        layers=config.payload_key_layers,
                        dropout=config.dropout,
                    ),
                )
                for payload_size in payload_sizes
            ]
        )
        self.value_networks = torch.nn.ModuleList(
            [
                FCEncoder(
                    payload_size // config.heads,
                    FCEncoderConfig(
                        hidden_size=config.payload_value_size,
                        output_size=config.payload_output_size,
                        layers=config.payload_value_layers,
                        dropout=config.dropout,
                    ),
                )
                for payload_size in payload_sizes
            ]
        )

        self.key_output_network = FCEncoder(
            config.payload_output_size,
            config=FCEncoderConfig(
                hidden_size=config.payload_output_size,
                output_size=config.payload_output_size // config.heads,
                layers=config.payload_output_layers,
                dropout=config.dropout,
            ),
        )
        self.value_output_network = FCEncoder(
            config.payload_output_size,
            config=FCEncoderConfig(
                hidden_size=config.payload_output_size,
                output_size=config.payload_output_size // config.heads,
                layers=config.payload_output_layers,
                dropout=config.dropout,
            ),
        )

    def _inject(
        self, input: torch.Tensor, payloads: typing.List[torch.Tensor]
    ) -> torch.Tensor:
        for payload in payloads:
            if payload.dim() != payloads[0].dim():
                raise ValueError("Payloads must have the same number of dimensions")

        dim = payloads[0].dim()

        batch_size = input.shape[0]
        utterances = input.shape[1] if dim > 2 else 1
        index = 2 if dim > 2 else 1

        attentions = []

        for i in range(self.heads):
            query = torch.matmul(
                input.view([batch_size * utterances] + list(input.shape[index:])),
                self.query_weight[i],
            )

            keys = [
                self.key_networks[p](
                    torch.matmul(
                        payload.view(
                            [batch_size * utterances] + list(payload.shape[index:])
                        ),
                        self.key_weights[p][i],
                    )
                )
                for p, payload in enumerate(payloads)
            ]
            values = [
                self.value_networks[p](
                    torch.matmul(
                        payload.view(
                            [batch_size * utterances] + list(payload.shape[index:])
                        ),
                        self.value_weights[p][i],
                    )
                )
                for p, payload in enumerate(payloads)
            ]

            key = torch.sigmoid(self.key_output_network(torch.prod(torch.stack(keys, dim=-1), dim=-1)))
            value = torch.sigmoid(self.value_output_network(torch.prod(torch.stack(values, dim=-1), dim=-1)))

            attention = torch.matmul(
                torch.softmax(
                    torch.matmul(query, key.transpose(-1, -2)) / self.scale,
                    dim=-1,
                ),
                value,
            )
            attentions.append(attention)

        return (
            torch.matmul(torch.cat(attentions, dim=-1), self.attention_weight)
            .view([batch_size, utterances] + list(input.shape[index:]))
            .squeeze()
        )

    def forward(
        self, input: torch.Tensor, payloads: typing.List[torch.Tensor]
    ) -> torch.Tensor:
        print(input.shape)

        x = self.dropout(input)

        for i, layer in enumerate(self.linear_layers):
            if i == 0:
                x = torch.tanh(layer(x))
            else:
                x = layer(x + self._inject(x, payloads))

        return torch.tanh(self.output_layer(x))
