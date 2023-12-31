import torch

class NCELoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer('avg_exp_score', torch.tensor(-1))
    
    def _compute_partition_function(self, predictions: torch.Tensor) -> torch.Tensor:
        if self.avg_exp_score > 0:
            return self.avg_exp_score

        with torch.no_grad():
            self.avg_exp_score = predictions.mean().unsqueeze(dim=0)
            return self.avg_exp_score

    def forward(self, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        K = negatives.shape[1]

        unnormalized_positives = torch.exp(positives)
        unnormalized_negatives = torch.exp(negatives)

        with torch.no_grad():
            unnormalized_average = self._compute_partition_function(unnormalized_negatives)
        
        Pmt = torch.div(unnormalized_positives, unnormalized_positives + K * unnormalized_average)
        lnPmtSum = -torch.log(Pmt).mean(dim=-1)

        Pon = torch.div(K * unnormalized_average, unnormalized_negatives + K * unnormalized_average)
        lnPonSum = -torch.log(Pon).sum(dim=-1)

        return (lnPmtSum + lnPonSum).mean()
            