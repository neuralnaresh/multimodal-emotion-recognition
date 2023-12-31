import math

import torch

from layers.config.fcinjection import FCInjectionConfig

class FCInjection(torch.nn.Module):
    def __init__(self, input_size: int, payload_size: int, config: FCInjectionConfig) -> None:
        super().__init__()

        if payload_size % config.heads != 0:
            raise ValueError("Injection payload size must be divisible by the number of attention heads")

        self.dropout = torch.nn.Dropout(p=config.dropout)
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(input_size if i == 0 else config.hidden_size, config.hidden_size) for i in range(config.layers + 1)])
        self.output_layer = torch.nn.Linear(config.hidden_size, config.output_size)

        self.heads = config.heads

        self.query_size = payload_size // config.heads
        self.key_size = payload_size // config.heads
        self.value_size = payload_size // config.heads

        self.scale = math.sqrt(float(self.query_size))

        self.query_weight = torch.nn.parameter.Parameter(torch.Tensor(self.heads, config.hidden_size, self.query_size))
        self.key_weight = torch.nn.parameter.Parameter(torch.Tensor(self.heads, payload_size, self.key_size))
        self.value_weight = torch.nn.parameter.Parameter(torch.Tensor(self.heads, payload_size, self.value_size))
        self.attention_weight = torch.nn.parameter.Parameter(torch.Tensor(self.heads * self.value_size, config.hidden_size))
        
    def _inject(self, input: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
        batch_size = input.shape[0]
        utterances = input.shape[1] if payload.dim() > 2 else 1
        index = 2 if payload.dim() > 2 else 1

        attentions = []

        for i in range(self.heads):
            query = torch.matmul(input.view([batch_size * utterances] + list(input.shape[index:])), self.query_weight[i])
            key = torch.matmul(payload.view([batch_size * utterances] + list(payload.shape[index:])), self.key_weight[i])
            value = torch.matmul(payload.view([batch_size * utterances] + list(payload.shape[index:])), self.value_weight[i])

            attention = torch.matmul(torch.softmax(torch.matmul(query, key.transpose(-1, -2)) / self.scale, dim=-1), value)
            attentions.append(attention)

        return torch.matmul(torch.cat(attentions, dim=-1), self.attention_weight).view([batch_size, utterances] + list(input.shape[index:])).squeeze()

    def forward(self, input, payload):
        x = self.dropout(input)

        for i, layer in enumerate(self.linear_layers):
            if i == 0:
                x = torch.tanh(layer(x))
            else:
                x = layer(x + self._inject(x, payload))
        
        return torch.tanh(self.output_layer(x))