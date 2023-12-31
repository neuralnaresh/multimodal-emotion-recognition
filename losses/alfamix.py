import torch

from losses.config.alfamix import ALFAMixLossConfig


class ALFAMixLoss(torch.nn.Module):
    def __init__(self, config: ALFAMixLossConfig) -> None:
        super().__init__()

        self.loss = torch.nn.CrossEntropyLoss(reduction=config.reduction.value)

    def forward(
        self,
        unfused_predictions: torch.Tensor,
        fused_predictions: torch.Tensor,
        _: torch.Tensor,
        __: torch.Tensor,
    ) -> torch.Tensor:
        return -self.loss(fused_predictions, torch.argmax(unfused_predictions, dim=-1))
