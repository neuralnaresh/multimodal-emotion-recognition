import typing

import torch
import torchaudio

import numpy as np

from dlpipeline.data.transform import Transformer

import data.constants
import data.utils

from data.transform.config.load_audio import LoadAudioConfig
from data.transform.config.load_video import LoadVideoConfig
from data.transform.config.extract_frames import ExtractFramesConfig

class LoadAudio(Transformer):
    def _sample_frame_indices(self, clip_len, seg_len):
        return np.linspace(0, seg_len - 1, clip_len).astype(np.int64)

    def _frames(self):
        return int(LoadVideoConfig().frames / ExtractFramesConfig().frame_rate * LoadAudioConfig().sample_rate)

    def metadata(self, meta: dict[str, typing.Any]) -> dict[str, typing.Any]:
        meta[data.constants.META_AUDIO_RAW_FRAMES] = self._frames()
        meta[data.constants.META_AUDIO_RAW_FRAME_SAMPLES] = LoadAudioConfig().sample_rate // ExtractFramesConfig().frame_rate

        return meta

    def transform_single(self, input_filename: str, _: str):
        sample = data.utils.read_pickle(input_filename)

        audio_path = sample[data.constants.DATA_AUDIO_PATH]
        audio = torch.nn.functional.normalize(torchaudio.load(audio_path, normalize=True)[0], dim=0).squeeze().numpy()

        audio_frames = audio.shape[0]
        target_frames = self._frames()

        if audio_frames < target_frames:
            audio = np.pad(audio, ((0, target_frames - audio_frames), (0, 0)), mode="constant")
        else:
            indices = self._sample_frame_indices(target_frames, audio_frames)
            audio = audio[indices]

        sample[data.constants.DATA_AUDIO_RAW] = np.expand_dims(audio, axis=-1)
        return sample

    def write_single(
        self, processed, input_filename: str, output_base_dir: str
    ) -> None:
        return data.utils.write_transformer_pickle(
            processed, input_filename, output_base_dir
        )
