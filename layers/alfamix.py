import typing

import torch

class ALFAMixFusion(torch.nn.Module):
    def __init__(
        self, input_size: int, modalities: int) -> None:
        super().__init__()

        self.alphas = torch.nn.ModuleList(
            [
                torch.nn.ParameterList(
                    [
                        torch.nn.parameter.Parameter(torch.empty(input_size))
                        for _ in range(modalities)
                    ]
                )
                for _ in range(modalities)
            ]
        )

        for i in range(modalities):
            for j in range(modalities):
                torch.nn.init.uniform_(self.alphas[i][j], 0, 1)

    def forward(self, encodings: typing.Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = {}

        keys = list(encodings.keys())

        for i in range(len(keys)):
            output = torch.zeros_like(encodings[keys[i]])

            for j in range(len(keys)):
                if i == j:
                    continue

                output += self.alphas[i][j] * encodings[keys[j]] + (1 - self.alphas[i][j]) * encodings[keys[i]]
            
            outputs[keys[i]] = output / (len(keys) - 1)

        return outputs                 