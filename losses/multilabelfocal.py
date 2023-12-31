import torch

from losses.config.reduction import Reduction
from losses.config.multilabelfocal import MultiLabelFocalLossConfig

class MultiLabelFocalLoss(torch.nn.Module):
    def __init__(self, config: MultiLabelFocalLossConfig) -> None:
        super().__init__()

        self.gamma = config.gamma
        self.reduction = config.reduction

        if isinstance(config.alpha, float):
            self.alpha = torch.Tensor([config.alpha, 1 - config.alpha])
        elif isinstance(config.alpha, list):
            self.alpha = torch.Tensor(config.alpha)
        else:
            self.alpha = None

    def forward(self, predictions: torch.Tensor, labels: torch.Tensor, _) -> torch.Tensor:
        p = torch.where(labels > 0.5, predictions, 1 - predictions)
        logp = -torch.log(torch.clamp(p, 1e-4, 1-1e-4))

        if self.alpha is not None:
            logp = logp * torch.autograd.Variable(self.alpha.type_as(logp))

        p = p.contiguous().view(-1)
        logp = logp.contiguous().view(-1)

        loss = ((1 - p) ** self.gamma) * logp

        if self.reduction == Reduction.MEAN:
            return p.shape[-1] * loss.mean()
        elif self.reduction == Reduction.SUM:
            return loss.sum()