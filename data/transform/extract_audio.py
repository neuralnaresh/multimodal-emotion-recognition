import os 
import subprocess

from dlpipeline.data.transform import Transformer

import data.constants
import data.utils

class ExtractAudio(Transformer):
    def _extract_audio(self, input_video: str, output_audio: str) -> None:
        result = subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', input_video, '-vn', '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '1', output_audio])
        result.check_returncode()

    def transform_single(self, input_filename: str, output_base_dir: str):
        data_sample = data.utils.read_pickle(input_filename)

        split = data.utils.split_from_filename(input_filename)
        video_id = data_sample[data.constants.DATA_ID]
        output_dir = f'{output_base_dir}/{split}/audio'
        output_audio = f'{output_dir}/{video_id}.wav'

        os.makedirs(output_dir, exist_ok=True)
        try:
            self._extract_audio(data_sample[data.constants.DATA_VIDEO_PATH], output_audio)
        except:
            return None

        data_sample[data.constants.DATA_AUDIO_PATH] = output_audio
        return data_sample

    def write_single(self, processed, input_filename: str, output_base_dir: str) -> None:
        data.utils.write_transformer_pickle(processed, input_filename, output_base_dir)