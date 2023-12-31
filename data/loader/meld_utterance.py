from dlpipeline.data.loader import Loader

import data.constants
import data.utils

class MELDUtteranceLoader(Loader):
    def __init__(self) -> None:
        super().__init__()

    def load_single(self, input_filename: str):
        return data.utils.read_pickle(input_filename)

    def post_load(self, data_list):
        data_list = data.utils.normalize_data_dict(data_list, data.constants.DATA_AUDIO_FEATURES)

        for item in data_list:
            item[data.constants.DATA_FRAME_SEQ_LENGTH] = item[data.constants.DATA_FRAME_EMBEDDING].shape[0]
            item[data.constants.DATA_AUDIO_SEQ_LENGTH] = item[data.constants.DATA_AUDIO_EMBEDDING].shape[0]

            if data.constants.DATA_PRIMARY_FACE_HOG in item:
                item[data.constants.DATA_FACE_FEATURES_SEQ_LENGTH] = item[data.constants.DATA_PRIMARY_FACE_HOG].shape[0]

        return data_list