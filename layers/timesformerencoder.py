import typing

import torch

from transformers import VideoMAEFeatureExtractor, VideoMAEModel

from layers.config.timesformerencoder import TimesformerEncoderConfig


class TimesformerEncoder(torch.nn.Module):
    def __init__(self, config: TimesformerEncoderConfig) -> None:
        super().__init__()

        self.dropout = torch.nn.Dropout(p=config.dropout)

        self.classifier = VideoMAEModel.from_pretrained(
            "MCG-NJU/videomae-large-finetuned-kinetics"
        )

        self.norm = torch.nn.LayerNorm(self.classifier.config.hidden_size)
        self.linear = torch.nn.Linear(self.classifier.config.hidden_size, config.output_size)

    def forward(self, videos: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        videos: (batch_size, frames, height, width, channels)
        lengths: (batch_size)
        """
        x = self.classifier(pixel_values=videos).last_hidden_state

        x = self.dropout(x)
        x = self.norm(x.mean(dim=1))
        x = self.linear(x)

        return torch.tanh(x)
