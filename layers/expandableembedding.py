import typing

import torch


def h_poly(t):
    tt = t[None, :] ** torch.arange(4, device=t.device)[:, None]
    A = torch.tensor(
        [[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]],
        dtype=t.dtype,
        device=t.device,
    )
    return A @ tt


def interp(x, y, xs):
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1]).unsqueeze(dim=-1)
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])

    idxs = torch.searchsorted(x[1:], xs)

    dx = x[idxs + 1] - x[idxs]
    hh = h_poly((xs - x[idxs]) / dx).unsqueeze(dim=-1)

    dx = dx.unsqueeze(dim=-1)

    return (
        hh[0] * y[idxs]
        + hh[1] * m[idxs] * dx
        + hh[2] * y[idxs + 1]
        + hh[3] * m[idxs + 1] * dx
    )


class ExpandableEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.max_buckets = num_embeddings
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)

    def _get_expanded_embeddings(self, max_target_buckets: int) -> torch.Tensor:
        available_buckets = torch.arange(self.max_buckets) / self.max_buckets
        available_buckets = available_buckets.to(torch.float32)

        query_buckets = torch.arange(max_target_buckets) / max_target_buckets
        query_buckets = query_buckets.to(torch.float32)

        available_embeddings = self.embedding.weight[Ellipsis]

        expanded_embeddings = interp(available_buckets, available_embeddings.to(available_buckets.device), query_buckets)

        return expanded_embeddings

    def forward(
        self,
        inputs: torch.Tensor,
        interpolate: bool = False,
        max_target_buckets: typing.Optional[int] = None,
    ) -> torch.Tensor:
        if interpolate:
            assert (
                max_target_buckets is not None
            ), "max_target_bucketes should be specified when interpolating"

            expanded_embeddings = self._get_expanded_embeddings(max_target_buckets)
            print(expanded_embeddings.shape)
            return torch.nn.functional.embedding(inputs, expanded_embeddings.to(inputs.device))
        else:
            return self.embedding(inputs)
