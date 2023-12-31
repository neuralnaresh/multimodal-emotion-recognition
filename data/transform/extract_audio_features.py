import typing

import torch
import opensmile
import torchaudio

import numpy as np

from hear21passt.base import load_model, get_scene_embeddings

from dlpipeline.data.transform import Transformer
from dlpipeline.commands.experiment_config import ExperimentConfig

import data.constants
import data.utils

from data.transform.config.extract_audio_features import ExtractAudioFeaturesConfig, AudioFeatureExtractor

class ExtractAudioFeatures(Transformer):
    def __init__(self) -> None:
        super().__init__()

        self._init_extractor()

    def _init_extractor(self) -> typing.Callable[[str], np.ndarray]:
        config = ExtractAudioFeaturesConfig()

        dimensions = {
            opensmile.FeatureSet.ComParE_2016: {
                opensmile.FeatureLevel.Functionals: 6373,
            },
        }

        if config.extractor == AudioFeatureExtractor.OPENSMILE:
            self.extractor = opensmile.Smile(
                feature_set=config.opensmile_featureset,
                feature_level=config.opensmile_featurelevel,
            )
            self.extract = lambda file: self.extractor.process_file(file).to_numpy().squeeze()
            self.feature_dimension = dimensions[config.opensmile_featureset][config.opensmile_featurelevel]
        elif config.extractor == AudioFeatureExtractor.PASST:
            def extract(file):
                audio, sample_rate = torchaudio.load(file)
                audio = torchaudio.functional.resample(audio, sample_rate, 32000)
                audio = audio.to(ExperimentConfig().device)
                
                return get_scene_embeddings(audio, self.extractor).detach().view(-1).cpu().numpy()

            self.extractor = load_model().to(ExperimentConfig().device)
            self.extract = extract
            self.feature_dimension = 1295 * 2
            
    def metadata(self, meta: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        meta[data.constants.META_AUDIO_FEATURE_DIM] = self.feature_dimension

        return meta

    def transform_single(self, input_filename: str, _: str):
        data_sample = data.utils.read_pickle(input_filename)

        result = self.extract(data_sample[data.constants.DATA_AUDIO_PATH])
        data_sample[data.constants.DATA_AUDIO_FEATURES] = result

        return data_sample

    def write_single(self, processed, input_filename: str, output_base_dir: str) -> None:
        data.utils.write_transformer_pickle(processed, input_filename, output_base_dir)
