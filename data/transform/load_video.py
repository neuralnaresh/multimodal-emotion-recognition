import os
import glob
import typing

import cv2

import numpy as np

from dlpipeline.data.transform import Transformer

import data.constants
import data.utils

from data.transform.config.load_video import LoadVideoConfig

class LoadVideo(Transformer):
    def _sample_frame_indices(self, clip_len, seg_len):
        return np.linspace(0, seg_len - 1, clip_len).astype(np.int64)

    def transform_single(self, input_filename: str, _: str) -> dict[str, typing.Any]:
        sample = data.utils.read_pickle(input_filename)
        
        frames_dir = sample[data.constants.DATA_FRAMES_DIR]

        frames = sorted(glob.glob(frames_dir + '/*.png'))
        images = [cv2.imread(frame) for frame in frames]

        if LoadVideoConfig().resize:
            images = [cv2.resize(image, (LoadVideoConfig().resize_width, LoadVideoConfig().resize_height)) for image in images]

        if (len(images) < LoadVideoConfig().frames):
            images = np.array(images + [images[-1]] * (LoadVideoConfig().frames - len(images)))
        else:
            indices = self._sample_frame_indices(LoadVideoConfig().frames, len(frames))
            images = np.array(images)[indices]

        sample[data.constants.DATA_VIDEO_RAW] = images.transpose(3, 0, 1, 2)

        return sample

    def write_single(self, processed: dict[str, typing.Any], input_filename: str, output_base_dir: str) -> None:
        return data.utils.write_transformer_pickle(processed, input_filename, output_base_dir)