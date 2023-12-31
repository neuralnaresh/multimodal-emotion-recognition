import typing

import torch
import torchvision

import numpy as np

from dlpipeline.data.loader import Loader

import data.constants
import data.utils

_T = dict[str, typing.Any]


class MMTLoader(Loader):
    def __init__(self) -> None:
        super().__init__()

        self._tokens: dict[str, int] = {"": 0}

        self.crop = torchvision.transforms.RandomResizedCrop(size=(224))
        self.flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        self.color = torchvision.transforms.ColorJitter(
            brightness=32 / 255, saturation=0.4, contrast=0.4, hue=0.4
        )

    def _get_token(self, word: str) -> int:
        word = (
            word.lower()
            .replace("'", "")
            .replace(",", "")
            .replace('"', "")
            .replace(".", "")
            .replace("-", "")
        )

        if word not in self._tokens:
            self._tokens[word] = len(self._tokens)

        return self._tokens[word]

    def load_single(self, input_filename: str) -> _T:
        return data.utils.read_pickle(input_filename)

    def post_load(self, items: list[_T]) -> list[_T]:
        for sample in items:
            if data.constants.DATA_TEXT in sample:
                tokens = np.array(
                    [
                        self._get_token(word)
                        for word in sample[data.constants.DATA_TEXT].split(" ")
                    ]
                )

                words = tokens.shape[0]

                if words < 32:
                    tokens = np.array(tokens.tolist() + [0] * (32 - words))
                else:
                    tokens = tokens[:32]

                sample[data.constants.DATA_TEXT_TOKENS] = tokens

        return items

    def pre_batch(self, sample: _T) -> _T:
        sample[data.constants.DATA_VIDEO_RAW] = torch.nn.functional.normalize(
            self.color(
                self.flip(
                    self.crop(torch.tensor(sample[data.constants.DATA_VIDEO_RAW]))
                ).transpose(0, 1)
            )
            .transpose(0, 1)
            .to(torch.float),
            dim=1,
        )

        return sample

    def pre_batch_eval(self, sample: _T) -> _T:
        sample[data.constants.DATA_VIDEO_RAW] = torch.nn.functional.normalize(
            self.crop(torch.tensor(sample[data.constants.DATA_VIDEO_RAW])).to(
                torch.float
            ),
            dim=1,
        )

        return sample
