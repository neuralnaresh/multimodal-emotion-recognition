import os
import typing

import torchvggish

import numpy as np

from dlpipeline.data.transform import Transformer
from dlpipeline.commands.experiment_config import ExperimentConfig

import data.constants
import data.utils

class EmbedAudio(Transformer):
    def __init__(self) -> None:
        super().__init__()

        self.model = self._create_model()

    def _create_model(self):
        model = torchvggish.vggish()
        
        # model = torch.nn.Sequential(*list(model.embeddings.children())[:-1])
        model.eval()

        return model

    def _embed_audio(self, audio_file: str):
        input =  torchvggish.vggish_input.wavfile_to_examples(audio_file)
        
        if input.shape[0] != 0:
            return self.model.forward(input).detach().numpy()
        else:
            return None

    def metadata(self, meta: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        meta[data.constants.META_AUDIO_EMBEDDING_DIM] = 128
        return meta

    def transform_single(self, input_filename: str, _: str):
        data_sample = data.utils.read_pickle(input_filename)

        if not os.path.exists(data_sample[data.constants.DATA_AUDIO_PATH]):
            return None

        embedding = self._embed_audio(data_sample[data.constants.DATA_AUDIO_PATH])

        if embedding is not None:
            if embedding.shape[0] == 128:
                embedding = np.expand_dims(embedding, axis=0)

            data_sample[data.constants.DATA_AUDIO_EMBEDDING] = embedding
            return data_sample
        else:
            return None
    
    def write_single(self, processed, input_filename: str, output_base_dir: str) -> None:
        data.utils.write_transformer_pickle(processed, input_filename, output_base_dir)