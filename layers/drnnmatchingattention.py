import torch


class DRNNMatchingAttention(torch.nn.Module):
    def __init__(self, memory_size: int, candidate_size: int) -> None:
        super().__init__()

        self.transform = torch.nn.Linear(candidate_size, memory_size, bias=True)

    def forward(
        self,
        memory: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        memory - (utterances, batch_size, memory_size)
        candidates - (batch_size, candidates_size)
        mask - (batch_size, utterances)
        """

        if mask is None:
            mask = torch.ones(memory.shape[0], memory.shape[1]).type(memory.type())

        alpha = torch.softmax(
            torch.bmm(
                self.transform(candidates).unsqueeze(dim=1), memory.permute(1, 2, 0)
            ),
            dim=2,
        )

        attention_pool = torch.bmm(alpha, memory.transpose(0, 1))[:, 0, :]

        return attention_pool, alpha
