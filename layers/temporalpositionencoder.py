import torch

from layers.expandableembedding import ExpandableEmbedding

from layers.config.temporalpositionencoder import TemporalPositionEncoderConfig

class TemporalPositionEncoder(torch.nn.Module):
    def __init__(self, config: TemporalPositionEncoderConfig) -> None:
        super().__init__()

        self.config = config

        self.temporal_position_embeddings = ExpandableEmbedding(
            config.max_temporal_buckets,
            config.hidden_size,
        )
        self.layernorm = torch.nn.LayerNorm(config.hidden_size, eps=config.epsilon)
        self.dropout = torch.nn.Dropout(config.dropout)

    def _lookup(self, ids: torch.Tensor, reference_buckets: int, target_buckets: int) -> torch.Tensor:
        if target_buckets == reference_buckets:
            embeddings = self.temporal_position_embeddings(ids, interpolate=False, max_target_buckets=None)
        else:
            embeddings = self.temporal_position_embeddings(ids, interpolate=True, max_target_buckets=target_buckets)

        return embeddings
    
    def forward(self, inputs: torch.Tensor, dimensions: list[int]) -> torch.Tensor:
        _, t, _ = dimensions
        temporal_positions_ids = torch.arange(t).to(inputs.device)

        embeddings = self._lookup(temporal_positions_ids, self.config.max_temporal_buckets, t)
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(inputs + embeddings)

        return embeddings