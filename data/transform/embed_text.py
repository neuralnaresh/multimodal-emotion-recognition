import typing

import torch

from sentence_transformers import SentenceTransformer

from dlpipeline.data.transform import Transformer
from dlpipeline.commands.experiment_config import ExperimentConfig

import data.constants
import data.utils

class EmbedText(Transformer):
    def __init__(self) -> None:
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=torch.device('cpu'))
    
    def metadata(self, meta: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        meta[data.constants.META_TEXT_EMBEDDING_DIM] = 768
        return meta

    def transform_single(self, input_filename: str, _: str):
        data_sample = data.utils.read_pickle(input_filename)

        text = data_sample[data.constants.DATA_TEXT]
        result = self.model.encode(text, convert_to_tensor=True).detach().cpu().squeeze().numpy()

        data_sample[data.constants.DATA_TEXT_EMBEDDING] = result

        return data_sample

    def write_single(self, processed, input_filename: str, output_base_dir: str) -> None:
        data.utils.write_transformer_pickle(processed, input_filename, output_base_dir)