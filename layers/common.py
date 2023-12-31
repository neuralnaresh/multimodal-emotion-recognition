import typing
import dataclasses

import torch

import numpy as np

@dataclasses.dataclass
class EncoderModuleBuild:
    size: int = None

    vertices: int = None
    channels: int = None
    persons: int = None
    adjacency_matrix: np.ndarray = None

@dataclasses.dataclass
class ModalityMeta:
    input_constructor: typing.Callable[[typing.Any], typing.Tuple[torch.Tensor]]
    build: EncoderModuleBuild
    dialog: bool = False

def pad_single(x: typing.Union[torch.tensor, torch.autograd.Variable], length: int):
    if length > x.shape[0]:
        return torch.cat([x, torch.zeros(length - x.shape[0], *x.shape[1:]).to(x.device)]).to(
            x.device
        )
    else:
        return x

def pad(tensors: typing.List[typing.Union[torch.Tensor, torch.autograd.Variable]], length: int) -> torch.Tensor:
    return torch.stack([pad_single(tensor, length) for tensor in tensors], dim=0)