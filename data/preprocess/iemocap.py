import os
import re
import glob
import math
import pickle
import random
import itertools
import subprocess
import typing

import numpy as np

from dlpipeline.data.preprocess import Preprocessor

import data.constants
import data.utils

_iemocap_data_type = typing.List[typing.Dict[str, typing.Any]]
_iemocap_class_map = {
    "neu": 0,
    "exc": 1,
    "fru": 2,
    "hap": 3,
    "sad": 4,
    "ang": 5,
    "fea": 6,
    "sur": 7,
    "dis": 8,
    "oth": 9,
    "xxx": 10,
}


class IEMOCAP(Preprocessor):
    def __init__(self) -> None:
        super().__init__()

        self.dataset_path = ""

    def load_data(self, dataset_path: str) -> typing.List[_iemocap_data_type]:
        self.dataset_path = dataset_path

        self.data = []

        for session in range(1, 6):
            dialogs = glob.glob(
                f"{dataset_path}/Session{str(session)}/dialog/transcriptions/*.txt"
            )

            for dialog in dialogs:
                with open(dialog, "r") as f:
                    for index, utterance in enumerate(f.readlines()):
                        match = re.match(
                            "(Ses0[0-9][MF]_[a-zA-Z]*[0-9]*_[FM][0-9]*) \[([0-9.\-]*)\]",
                            utterance,
                        )

                        if match is not None:
                            self.data.append(
                                {
                                    data.constants.IEMOCAP_DIALOG_ID: "_".join(
                                        match[1].split("_")[:-1]
                                    ),
                                    data.constants.IEMOCAP_UTTERANCE_ID: match[1],
                                    data.constants.IEMOCAP_UTTERANCE_INDEX: index,
                                    data.constants.IEMOCAP_SESSION: session,
                                    data.constants.IEMOCAP_UTTERANCE_START: round(
                                        float(match[2].split("-")[0]), 4
                                    ),
                                    data.constants.IEMOCAP_UTTERANCE_END: round(
                                        float(match[2].split("-")[1]), 4
                                    ),
                                    data.constants.DATA_ID: match[1],
                                    data.constants.DATA_TEXT: utterance.split("]:")[-1],
                                }
                            )

        return self.data

    def split_data(self) -> None:
        self.data.sort(key=lambda x: x[data.constants.IEMOCAP_UTTERANCE_ID])

        dialogs = []

        for _, group in itertools.groupby(self.data, lambda x: x[data.constants.IEMOCAP_DIALOG_ID]):
            dialogs.append(list(group))

        random.shuffle(dialogs)

        train = math.floor(self.split.train / 100 * len(dialogs))
        test = math.floor(self.split.test / 100 * len(dialogs))

        self.train = dialogs[:train]
        self.test = dialogs[train : train + test]
        self.validation = dialogs[train + test :]

        self.train = [utterance for dialog in self.train for utterance in dialog]
        self.test = [utterance for dialog in self.test for utterance in dialog]
        self.validation = [utterance for dialog in self.validation for utterance in dialog]

    def _convert_timestamp(self, timestamp: float) -> float:
        return (timestamp + 2) / 100

    def _split_video(
        self, item: _iemocap_data_type, output_dir: str, split: str
    ) -> str:
        os.makedirs(f"{output_dir}/{split}/videos/", exist_ok=True)

        input_video = f"{self.dataset_path}/Session{item[data.constants.IEMOCAP_SESSION]}/dialog/avi/DivX/{item[data.constants.IEMOCAP_DIALOG_ID]}.avi"
        output_video = f"{output_dir}/{split}/videos/{item[data.constants.IEMOCAP_UTTERANCE_ID]}.avi"

        result = subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-hide_banner",
                "-ss",
                str(item[data.constants.IEMOCAP_UTTERANCE_START]),
                "-to",
                str(item[data.constants.IEMOCAP_UTTERANCE_END]),
                "-y",
                "-i",
                input_video,
                output_video,
            ]
        )
        result.check_returncode()

        return output_video

    def _set_evaluations(self, item: _iemocap_data_type) -> _iemocap_data_type:
        with open(
            f"{self.dataset_path}/Session{item[data.constants.IEMOCAP_SESSION]}/dialog/EmoEvaluation/{item[data.constants.IEMOCAP_DIALOG_ID]}.txt",
            "r",
        ) as f:
            for line in f.readlines():
                match = re.match(
                    f"\[[0-9.]* - [0-9.]*\][ \t]*{item[data.constants.IEMOCAP_UTTERANCE_ID]}[ \t]*([a-z]{{3}})[ \t]*\[([0-9. ,]*)\]",
                    line,
                )

                if match is not None:
                    item[data.constants.DATA_LABEL] = data.utils.one_hot_encode(
                        _iemocap_class_map[match[1]], len(_iemocap_class_map)
                    )

                    item[data.constants.DATA_VALENCE] = float(match[2].split(",")[0])
                    item[data.constants.DATA_AROUSAL] = float(match[2].split(",")[1])
                    item[data.constants.DATA_DOMINANCE] = float(match[2].split(",")[2])

                    return item

        raise ValueError(
            f"Could not find evaluations for utterance with id {item[data.constants.IEMOCAP_UTTERANCE_ID]}"
        )

    def _get_audio_path(self, item: _iemocap_data_type) -> str:
        return f"{self.dataset_path}/Session{item[data.constants.IEMOCAP_SESSION]}/sentences/wav/{item[data.constants.IEMOCAP_DIALOG_ID]}/{item[data.constants.IEMOCAP_UTTERANCE_ID]}.wav"

    def _get_speaker(self, item: _iemocap_data_type) -> np.ndarray:
        speaker_map = {"M": 0, "F": 1}

        return data.utils.one_hot_encode(
            speaker_map[item[data.constants.IEMOCAP_UTTERANCE_ID].split("_")[-1][0]], 2
        )

    def _get_mocap_connections(self) -> typing.List[typing.List[int]]:
        # [00:03] - Chin
        # [03:06] - Forehead
        # [06:14] - Left Cheek
        # [14:22] - Right Cheek
        # [22:23] - Left Eyelid
        # [23:24] - Right Eyelid
        # [24:29] - Nose
        # [29:33] - LBM
        # [33:37] - RBM
        # [37:41] - Left Eyebrow
        # [41:45] - Right Eyebrow
        # [45:53] - Mouth
        # [53:54] - LHD
        # [54:55] - RHD

        connections = [(i, i + 1) for i in range(55)]
        connection_pieces = [0, 3, 6, 14, 22, 23, 24, 29, 33, 37, 41, 45, 53, 54, 55]

        inward_connections = [
            connection
            for i in range(len(connection_pieces) - 1)
            for connection in connections[
                connection_pieces[i] : connection_pieces[i + 1] - 1
            ]
        ]
        outward_connections = [(j, i) for i, j in inward_connections]
        self_connections = [(i, i) for i in range(55)]

        additional_inward_connections = [
            (24, 25),
            (25, 26),
            (25, 27),
            (25, 28),  # nose
            (17, 19),
            (16, 18),
            (15, 18),
            (18, 20),
            (19, 21),
            (16, 20),
            (14, 17),  # right cheek
            (8, 10),
            (10, 12),
            (11, 13),
            (8, 12),
            (7, 10),
            (9, 11),
            (6, 9),  # left cheek
        ]
        additional_outward_connections = [
            (j, i) for i, j in additional_inward_connections
        ]

        return data.utils.edge_list_to_adjacency_matrix(
            inward_connections
            + outward_connections
            + self_connections
            + additional_inward_connections
            + additional_outward_connections,
            55,
        ).tolist()

    def _get_mocap_data(self, item: _iemocap_data_type) -> np.ndarray:
        with open(
            f"{self.dataset_path}/Session{item[data.constants.IEMOCAP_SESSION]}/sentences/MOCAP_rotated/{item[data.constants.IEMOCAP_DIALOG_ID]}/{item[data.constants.IEMOCAP_UTTERANCE_ID]}.txt",
            "r",
        ) as f:
            lines = f.readlines()[2:]

            mocap = np.ndarray((len(lines), 55, 3), dtype=np.float32)

            for i, line in enumerate(lines):
                coords = [float(coord) for coord in line.split(" ")][2:]
                mocap[i] = np.array(coords).reshape((55, 3))

        return mocap

    def metadata(self) -> typing.Dict[str, typing.Any]:
        return {
            data.constants.META_N_TRAIN: len(self.train),
            data.constants.META_N_VALIDATION: len(self.validation),
            data.constants.META_N_TEST: len(self.test),
            data.constants.META_CLASSES: _iemocap_class_map,
            data.constants.META_N_CLASSES: len(_iemocap_class_map),
            data.constants.META_CONVERSATIONAL: True,
            data.constants.META_MOCAP_DIM: 55,
            data.constants.META_MOCAP_COORD_DIM: 3,
            data.constants.META_MOCAP_CONNECTIONS: self._get_mocap_connections(),
        }

    def preprocess_single(
        self, item: _iemocap_data_type, base_dir: str, split: str
    ) -> _iemocap_data_type:
        try:
            item[data.constants.DATA_VIDEO_PATH] = self._split_video(item, base_dir, split)
            item[data.constants.DATA_AUDIO_PATH] = self._get_audio_path(item)
            item[data.constants.DATA_SPEAKER] = self._get_speaker(item)
            item[data.constants.DATA_FACE_MOCAP] = self._get_mocap_data(item)
            item[data.constants.DATA_FACE_MOCAP_SEQ_LENGTH] = item[
                data.constants.DATA_FACE_MOCAP
            ].shape[0]
        except FileNotFoundError as e:
            print(e)
            exit()

        return self._set_evaluations(item)

    def write_single(
        self, _, processed: _iemocap_data_type, base_dir: str, split: str
    ) -> None:
        if processed is not None:
            filename = (
                f"{base_dir}/{split}/{processed[data.constants.IEMOCAP_UTTERANCE_ID]}.pkl"
            )
            with open(filename, "wb") as f:
                pickle.dump(processed, f)
