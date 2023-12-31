import os
import subprocess
import typing

from dlpipeline.data.transform import Transformer

import data.constants
import data.utils

from data.transform.config.extract_frames import ExtractFramesConfig

class ExtractFrames(Transformer):
    def _split_frames(self, input_video: str, output_dir: str, frame_rate: int) -> None:
        os.makedirs(output_dir, exist_ok=True)
        result = subprocess.run(['ffmpeg', '-loglevel', 'error', '-hide_banner', '-i', input_video, '-vf', f'fps={frame_rate}', f'{os.path.abspath(output_dir)}/%04d.png'])
        result.check_returncode()

    def metadata(self, meta: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        meta[data.constants.META_FRAME_RATE] = ExtractFramesConfig().frame_rate
        return meta

    def transform_single(self, input_filename: str, output_base_dir: str):
        data_sample = data.utils.read_pickle(input_filename)

        split = data.utils.split_from_filename(input_filename)
        video_id = data_sample[data.constants.DATA_ID]
        output_dir = f'{output_base_dir}/{split}/frames/{video_id}'

        self._split_frames(data_sample[data.constants.DATA_VIDEO_PATH], output_dir, ExtractFramesConfig().frame_rate)

        data_sample[data.constants.DATA_FRAMES_DIR] = output_dir

        return data_sample

    def write_single(self, processed, input_filename: str, output_base_dir: str) -> None:
        data.utils.write_transformer_pickle(processed, input_filename, output_base_dir)