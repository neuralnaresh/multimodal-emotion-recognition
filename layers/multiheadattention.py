import typing

import torch


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        key_value_size: int,
        heads: int,
        bias: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.key_value_size = key_value_size
        self.heads = heads
        self.bias = bias

        self.inner_size = self.heads * self.key_value_size

        self.query = torch.nn.Linear(self.input_size, self.inner_size, bias=self.bias)
        self.key = torch.nn.Linear(self.input_size, self.inner_size, bias=self.bias)
        self.value = torch.nn.Linear(self.input_size, self.inner_size, bias=self.bias)

        self.output = torch.nn.Linear(self.inner_size, self.output_size, bias=self.bias)

        self.dropout = torch.nn.Dropout(dropout)

    def _split_heads(self, input: torch.Tensor, batch_size: int) -> torch.Tensor:
        return input.view(batch_size, -1, self.heads, self.key_value_size).permute(
            0, 2, 1, 3
        )

    def _join_heads(self, input: torch.Tensor, batch_size: int) -> torch.Tensor:
        return input.permute(0, 2, 1, 3).reshape(batch_size, -1, self.inner_size)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: typing.Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = query.shape[0]

        q = self.query(query)
        k = self.key(key)
        v = self.value(value)

        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        scores = torch.einsum("bnqd,bnkd->bnqk", q, k) / torch.sqrt(
            torch.tensor(k.shape[-1]).to(k.dtype)
        )

        if mask is not None:
            scores += mask

        attention = torch.nn.functional.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        hidden = attention @ v
        hidden = self._join_heads(hidden, batch_size)
        hidden = self.output(hidden)

        return {
            "hidden": hidden,
            "attention": attention
        }