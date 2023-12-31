import os
import glob
import typing
import subprocess

import h5py

import pandas as pd
import numpy as np

from dlpipeline.data.preprocess import Preprocessor

import data.constants
import data.utils

_mosei_data_type = typing.List[typing.Dict[str, typing.Any]]
_mosei_class_map = {
    "happy": 0,
    "sad": 1,
    "anger": 2,
    "surprise": 3,
    "disgust": 4,
    "fear": 5,
}


class MOSEI(Preprocessor):
    def __init__(self) -> None:
        super().__init__()

    def _split_video(self, item: _mosei_data_type, output_dir: str, split: str) -> str:
        os.makedirs(os.path.join(output_dir, split, 'videos', item[data.constants.MOSEI_ID]), exist_ok=True)

        input_video = item[data.constants.MOSEI_VIDEO_PATH]
        output_video = os.path.join(output_dir, split, 'videos', item[data.constants.MOSEI_ID], f'{item[data.constants.MOSEI_SEGMENT]}.avi')

        result = subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-hide_banner",
                "-ss",
                str(item[data.constants.MOSEI_SEGMENT_START]),
                "-to",
                str(item[data.constants.MOSEI_SEGMENT_END]),
                "-y",
                "-i",
                input_video,
                output_video,
            ]
        )
        result.check_returncode()

        return output_video

    def load_data(self, dataset_path: str) -> None:
        dataset = pd.read_csv(os.path.join(dataset_path, "mosei.csv"))
        words = h5py.File(
            os.path.join(dataset_path, "CMU_MOSEI_TimestampedWords.csd"), "r"
        )

        samples = []

        for index, row in dataset.iterrows():
            id = row["id"]
            segment = row["segment_id"]
            link = row["link"]
            start = row["start"]
            end = row["end"]

            if not os.path.exists(os.path.join(dataset_path, "videos", id)):
                continue

            video = list(glob.glob(os.path.join(dataset_path, "videos", id, "*.mp4")))
            if not len(video) == 1:
                continue

            video = video[0]

            text = " ".join(
                [
                    word[0].decode('utf-8')
                    for index, word in enumerate(words["words"]["data"][id]["features"])
                    if start <= words["words"]["data"][id]["intervals"][index][0] <= end
                ]
            ).replace(" sp", "")

            sample = {}

            sample[data.constants.MOSEI_ID] = id
            sample[data.constants.MOSEI_SEGMENT] = segment
            sample[data.constants.MOSEI_LINK] = link
            sample[data.constants.MOSEI_SEGMENT_START] = start
            sample[data.constants.MOSEI_SEGMENT_END] = end
            sample[data.constants.MOSEI_VIDEO_PATH] = video

            sample[data.constants.DATA_ID] = f'{id}_{segment}'
            sample[data.constants.DATA_INDEX] = index
            sample[data.constants.DATA_TEXT] = text
            sample[data.constants.DATA_LABEL] = np.array(row.iloc[6:12].tolist())

            samples.append(sample)

        self.data = samples

    def metadata(self) -> typing.Dict[str, typing.Any]:
        return {
            data.constants.META_N_TRAIN: len(self.train),
            data.constants.META_N_VALIDATION: len(self.validation),
            data.constants.META_N_TEST: len(self.test),
            data.constants.META_CLASSES: _mosei_class_map,
            data.constants.META_N_CLASSES: len(_mosei_class_map),
            data.constants.META_CONVERSATIONAL: False,
            data.constants.META_MULTICLASS: True,
        }
    
    def preprocess_single(self, item: _mosei_data_type, base_dir: str, split: str) -> _mosei_data_type:
        item[data.constants.DATA_VIDEO_PATH] = self._split_video(item, base_dir, split)

        return item
    
    def write_single(self, _: _mosei_data_type, processed: _mosei_data_type, base_dir: str, split: str) -> None:
        data.utils.write_pickle(processed, f"{base_dir}/{split}/{processed[data.constants.DATA_ID]}.pkl")
    
