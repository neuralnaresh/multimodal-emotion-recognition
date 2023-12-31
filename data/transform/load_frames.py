import os
import glob

import cv2

import numpy as np 

from transformers import VideoMAEFeatureExtractor

from dlpipeline.data.transform import Transformer

import data.constants
import data.utils

from data.transform.config.load_frames import LoadFramesConfig, FramesSource

class LoadFrames(Transformer):
    def __init__(self):
        super().__init__()

        self.processor = VideoMAEFeatureExtractor.from_pretrained(
            "MCG-NJU/videomae-large-finetuned-kinetics"
        )

    def _sample_frame_indices(self, clip_len, seg_len):
        return np.linspace(0, seg_len - 1, clip_len).astype(np.int64)

    def _process(self, images):
        return self.processor(images, return_tensors="np")

    def transform_single(self, input_filename: str, output_base_dir: str):
        data_sample = data.utils.read_pickle(input_filename)
        
        if LoadFramesConfig().source == FramesSource.FRAME:
            frames_dir = data_sample[data.constants.DATA_FRAMES_DIR]
        elif LoadFramesConfig().source == FramesSource.FACE:
            frames_dir = data_sample[data.constants.DATA_FRAME_FACES_DIR]
        elif LoadFramesConfig().source == FramesSource.BG:
            frames_dir = data_sample[data.constants.DATA_FRAME_BG_DIR]

        frames = sorted(glob.glob(frames_dir + '/*.png'))
        images = [cv2.imread(frame) for frame in frames]

        if (len(images) < LoadFramesConfig().frames):
            images = images + [images[-1]] * (LoadFramesConfig().frames - len(images))
        else:
            indices = self._sample_frame_indices(LoadFramesConfig().frames, len(frames))
            images = np.array(images)[indices]
            images = [images[i] for i in range(images.shape[0])]

        images = self._process(images)['pixel_values'][0]

        data_sample[data.constants.DATA_FRAMES] = images

        data.utils.write_transformer_pickle(data_sample, input_filename, output_base_dir)

        return None

    def write_single(self, processed, input_filename: str, output_base_dir: str) -> None:
        return