import pickle
import typing

import pandas as pd

from dlpipeline.data.preprocess import Preprocessor

import data.constants
import data.utils

_meld_data_type = typing.List[typing.Dict[str, typing.Any]]
_meld_class_map = {
    "neutral": 0,
    "surprise": 1,
    "fear": 2,
    "sadness": 3,
    "joy": 4,
    "disgust": 5,
    "anger": 6,
}

class MELD(Preprocessor):
    def __init__(self) -> None:
        super().__init__()

        self._dataset_path = ''
        self._meld_speaker_map = {}

    def load_data(self, dataset_path: str) -> None:
        self._dataset_path = dataset_path

        train = pd.read_csv(dataset_path + "/train.csv")
        dev = pd.read_csv(dataset_path + "/dev.csv")
        test = pd.read_csv(dataset_path + "/test.csv")

        speakers = []

        speakers.extend(train[data.constants.MELD_SPEAKER].unique())
        speakers.extend(dev[data.constants.MELD_SPEAKER].unique())
        speakers.extend(test[data.constants.MELD_SPEAKER].unique())

        self._meld_speaker_map = {speaker: i for i, speaker in enumerate(set(speakers))}

        self.train = train.to_dict("records")
        self.validation = dev.to_dict("records")
        self.test = test.to_dict("records")

    def split_data(self) -> None:
        return

    def _clean(self, sentence: str) -> str:
        sentence = sentence.replace("\x92", "'")
        sentence = sentence.replace("\x97", " ")
        sentence = sentence.replace("\x91", "'")
        sentence = sentence.replace("\x85", "")
        sentence = sentence.replace("\x93", '"')

        return sentence.lower()

    def metadata(self) -> typing.Dict[str, typing.Any]:
        return {
            data.constants.META_N_TRAIN: len(self.train),
            data.constants.META_N_VALIDATION: len(self.validation),
            data.constants.META_N_TEST: len(self.test),
            data.constants.META_CLASSES: _meld_class_map,
            data.constants.META_N_CLASSES: len(_meld_class_map),
            data.constants.META_SPEAKERS: self._meld_speaker_map,
            data.constants.META_N_SPEAKERS: len(self._meld_speaker_map),
            data.constants.META_CONVERSATIONAL: True
        }

    def preprocess_single(self, item: _meld_data_type, _: str, split: str) -> _meld_data_type:
        item[data.constants.MELD_UTTERANCE] = self._clean(item[data.constants.MELD_UTTERANCE])
        item[data.constants.MELD_VIDEO_ID] = f"dia{item[data.constants.MELD_DIALOG_ID]}_utt{item[data.constants.MELD_UTTERANCE_ID]}"

        item[data.constants.DATA_ID] = item[data.constants.MELD_VIDEO_ID]
        item[data.constants.DATA_LABEL] = data.utils.one_hot_encode(_meld_class_map[item[data.constants.MELD_EMOTION]], 7)
        item[data.constants.DATA_TEXT] = item[data.constants.MELD_UTTERANCE]
        item[data.constants.DATA_VIDEO_PATH] = f"{self._dataset_path}/{split}/{item[data.constants.MELD_VIDEO_ID]}.mp4"
        item[data.constants.DATA_SPEAKER] = data.utils.one_hot_encode(self._meld_speaker_map[item[data.constants.MELD_SPEAKER]], len(self._meld_speaker_map))

        return item

    def write_single(
        self, _: _meld_data_type, processed: _meld_data_type, base_dir: str, split: str
    ) -> None:
        filename = f"{base_dir}/{split}/{processed[data.constants.MELD_VIDEO_ID]}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(processed, f)
