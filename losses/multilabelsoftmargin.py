import torch

from losses.config.multilabelsoftmargin import MultiLabelSoftMarginLossConfig

class MultiLabelSoftMarginLoss(torch.nn.Module):
    def __init__(self, config: MultiLabelSoftMarginLossConfig) -> None:
        super().__init__()

        self.loss = torch.nn.MultiLabelSoftMarginLoss(weight=config.weight, reduction=config.reduction.value)

    def forward(self, prediction: torch.Tensor, labels: torch.Tensor, _) -> torch.Tensor:
        return self.loss(prediction, labels)
