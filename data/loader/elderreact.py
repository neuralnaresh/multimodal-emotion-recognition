from dlpipeline.data.loader import Loader

import numpy as np

import data.constants
import data.utils

class ElderReactLoader(Loader):
    def __init__(self) -> None:
        super().__init__()

    def load_single(self, input_filename: str):
        return data.utils.read_pickle(input_filename)

    def post_load(self, data_list):
        if data.constants.DATA_AUDIO_FEATURES in data_list[0]:
            data_list = data.utils.normalize_data_dict(data_list, data.constants.DATA_AUDIO_FEATURES)

        for i, item in enumerate(data_list):
            if data.constants.DATA_PRIMARY_FACE_HOG in item:
                item[data.constants.DATA_FACE_FEATURES_SEQ_LENGTH] = np.array(item[data.constants.DATA_PRIMARY_FACE_HOG]).shape[0]
            elif data.constants.DATA_PRIMARY_FACE_AUS in item:
                item[data.constants.DATA_FACE_FEATURES_SEQ_LENGTH] = item[data.constants.DATA_PRIMARY_FACE_AUS].shape[0]
            elif data.constants.DATA_PRIMARY_FACE_LANDMARKS in item:
                item[data.constants.DATA_FACE_FEATURES_SEQ_LENGTH] = item[data.constants.DATA_PRIMARY_FACE_LANDMARKS].shape[0]
            
            if data.constants.DATA_AUDIO in item:
                item[data.constants.DATA_AUDIO_SEQ_LENGTH] = item[data.constants.DATA_AUDIO].shape[0]

            item[data.constants.DATA_INDEX] = i

        return data_list