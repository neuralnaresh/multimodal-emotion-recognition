import typing

import torch

from layers.expandableembedding import ExpandableEmbedding

from layers.config.spatiotemporalpositionencoder import (
    SpatioTemporalPositionEncoderConfig,
)


class SpatioTemporalPositionEncoder(torch.nn.Module):
    def __init__(self, config: SpatioTemporalPositionEncoderConfig) -> None:
        super().__init__()

        self.config = config

        self.temporal_position_embeddings = ExpandableEmbedding(
            config.max_temporal_buckets, config.hidden_size
        )
        self.vertical_position_embeddings = ExpandableEmbedding(
            config.max_vertical_buckets, config.hidden_size
        )
        self.horizontal_position_embeddings = ExpandableEmbedding(
            config.max_horizontal_buckets, config.hidden_size
        )

        self.layernorm = torch.nn.LayerNorm(config.hidden_size, eps=config.epsilon)
        self.dropout = torch.nn.Dropout(config.dropout)

    def _build_video_ids(self, t: int, h: int, w: int) -> torch.Tensor:
        temporal_ids = torch.arange(t)[:, None, None]
        vertical_ids = torch.arange(h)[None, :, None]
        horizontal_ids = torch.arange(w)[None, None, :]

        temporal_ids = torch.tile(temporal_ids, (1, h, w))
        vertical_ids = torch.tile(vertical_ids, (t, 1, w))
        horizontal_ids = torch.tile(horizontal_ids, (t, h, 1))

        positional_ids = torch.stack(
            [temporal_ids, vertical_ids, horizontal_ids], dim=-1
        )
        positional_ids = positional_ids.view(-1, 3)

        return positional_ids

    def _lookup(
        self,
        lookup: typing.Callable[
            [torch.Tensor, typing.Optional[bool], typing.Optional[int]], torch.Tensor
        ],
        keys: torch.Tensor,
        reference_buckets: int,
        target_buckets: int,
    ) -> torch.Tensor:
        if target_buckets == reference_buckets:
            embeddings = lookup(keys, interpolate=False, max_target_buckets=None)
        else:
            embeddings = lookup(
                keys, interpolate=True, max_target_buckets=target_buckets
            )

        return embeddings

    def forward(self, inputs: torch.Tensor, dimensions: list[int]) -> torch.Tensor:
        _, t, h, w, _ = dimensions
        positional_ids = self._build_video_ids(t, h, w).to(inputs.device)

        temporal_position_ids = positional_ids[None, :, 0]
        vertical_position_ids = positional_ids[None, :, 1]
        horizontal_position_ids = positional_ids[None, :, 2]

        temporal_position_embeddings = self._lookup(
            self.temporal_position_embeddings,
            temporal_position_ids,
            self.config.max_temporal_buckets,
            t,
        )
        vertical_position_embeddings = self._lookup(
            self.vertical_position_embeddings,
            vertical_position_ids,
            self.config.max_vertical_buckets,
            h,
        )
        horizontal_position_embeddings = self._lookup(
            self.horizontal_position_embeddings,
            horizontal_position_ids,
            self.config.max_horizontal_buckets,
            w,
        )

        positional_embeddings = (
            temporal_position_embeddings
            + vertical_position_embeddings
            + horizontal_position_embeddings
        )
        positional_embeddings = self.layernorm(positional_embeddings)
        positional_embeddings = self.dropout(inputs + positional_embeddings)

        return positional_embeddings
