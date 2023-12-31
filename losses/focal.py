import torch

from losses.config.reduction import Reduction
from losses.config.focal import FocalLossConfig

class FocalLoss(torch.nn.Module):
    def __init__(self, config: FocalLossConfig) -> None:
        super().__init__()

        self.gamma = config.gamma
        self.reduction = config.reduction

        if isinstance(config.alpha, float):
            self.alpha = torch.Tensor([config.alpha, 1 - config.alpha])
        elif isinstance(config.alpha, list):
            self.alpha = torch.Tensor(config.alpha)
        else:
            self.alpha = None
    
    def forward(self, predictions: torch.Tensor, labels: torch.Tensor, utterance_lengths: torch.Tensor) -> torch.Tensor:
        if utterance_lengths is not None:
            labels = torch.argmax(labels, dim=-1)
            labels = torch.cat(
                [
                    labels[j][: utterance_lengths[j]]
                    for j in range(labels.shape[0])
                ]
            )
            labels = labels.contiguous().view(-1, 1)

            logpt = predictions.gather(1, labels.long()).view(-1)
            pt = torch.autograd.Variable(logpt.data.exp())

            if self.alpha is not None:
                alpha = self.alpha.type_as(predictions.type())
                alpha = alpha.gather(0, labels.data.view(-1))
                logpt = logpt * torch.autograd.Variable(alpha)

            loss = -1 * (1 - pt) ** self.gamma * logpt
        else:
            p = predictions.gather(1, labels.long()).view(-1)
            logp = -torch.log(torch.clamp(p, 1e-4, 1-1e-4))

            if self.alpha is not None:
                logp = logp * torch.autograd.Variable(self.alpha.type_as(logp))

            p = p.contiguous().view(-1)
            logp = logp.contiguous().view(-1)

            loss = -1 * ((1 - p) ** self.gamma) * logp

        if self.reduction == Reduction.MEAN:
            return loss.mean()
        elif self.reduction == Reduction.SUM:
            return loss.sum()