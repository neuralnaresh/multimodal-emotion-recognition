import typing

import pandas as pd

from dlpipeline.data.preprocess import Preprocessor

import data.constants
import data.utils

_elderreact_data_type = typing.List[typing.Dict[str, typing.Any]]
_elderreact_class_map = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happiness": 3,
    "sadness": 4,
    "surprise": 5,
}


class MELD(Preprocessor):
    def __init__(self) -> None:
        super().__init__()

        self._dataset_path = ""

    def _read_data(self, file: str) -> pd.DataFrame:
        df = pd.read_csv(file, sep=" ", header=None)
        df[data.constants.ELDERREACT_LABEL] = df.iloc[:, 1:7].values.tolist()
        df = df.iloc[:, [0, 7, 8, 9]]
        df.columns = [
            data.constants.ELDERREACT_ID,
            data.constants.ELDERREACT_GENDER,
            data.constants.ELDERREACT_VALENCE,
            data.constants.ELDERREACT_LABEL,
        ]
        return df

    def _get_data(self, split: str) -> typing.List[_elderreact_data_type]:
        return self._read_data(f"{self._dataset_path}/{split}_labels.txt").to_dict(
            "records"
        )

    def load_data(self, dataset_path: str) -> None:
        self._dataset_path = dataset_path

        self.train = self._get_data("train")
        self.validation = self._get_data("dev")
        self.test = self._get_data("test")

    def split_data(self) -> None:
        return

    def metadata(self) -> typing.Dict[str, typing.Any]:
        return {
            data.constants.META_N_TRAIN: len(self.train),
            data.constants.META_N_VALIDATION: len(self.validation),
            data.constants.META_N_TEST: len(self.test),
            data.constants.META_CLASSES: _elderreact_class_map,
            data.constants.META_N_CLASSES: len(_elderreact_class_map),
            data.constants.META_CONVERSATIONAL: False,
            data.constants.META_MULTICLASS: True,
        }

    def preprocess_single(
        self, item: _elderreact_data_type, _: str, split: str
    ) -> _elderreact_data_type:
        item[data.constants.DATA_ID] = item[data.constants.ELDERREACT_ID].split(".")[0]
        item[data.constants.DATA_LABEL] = item[data.constants.ELDERREACT_LABEL]
        item[
            data.constants.DATA_VIDEO_PATH
        ] = f"{self._dataset_path}/{split}/{item[data.constants.ELDERREACT_ID]}"

        return item

    def write_single(
        self,
        _: _elderreact_data_type,
        processed: _elderreact_data_type,
        base_dir: str,
        split: str,
    ) -> None:
        data.utils.write_pickle(
            processed,
            f"{base_dir}/{split}/{processed[data.constants.DATA_ID]}.pkl",
        )
