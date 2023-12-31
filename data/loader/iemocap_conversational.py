import pickle
import typing
import itertools

import numpy as np

from dlpipeline.data.loader import Loader

import data.constants
import data.utils

class IEMOCAPConversationalLoader(Loader):
    def __init__(self) -> None:
        super().__init__()

        self.max_utterance_length = 0

    def meta(self, meta: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        print('asdasd', self.max_utterance_length)
        exit()
        meta[data.constants.META_MAX_UTTERANCE_LENGTH] = self.max_utterance_length
        return meta

    def load_single(self, input_filename: str):
        return data.utils.read_pickle(input_filename)

    def post_load(self, data_list):
        data_list = data.utils.normalize_data_dict(data_list, data.constants.DATA_AUDIO_FEATURES)
        
        grouped_data = []

        data_list.sort(key=lambda x: x[data.constants.IEMOCAP_DIALOG_ID])

        for _, dialog_data in itertools.groupby(data_list, lambda x: x[data.constants.IEMOCAP_DIALOG_ID]):
            dialog_data = list(dialog_data)
            
            dialog_speakers = {x: i for i, x in enumerate(set([np.argmax(x[data.constants.DATA_SPEAKER]) for x in dialog_data]))}

            for item in dialog_data:
                if data.constants.DATA_FRAME_EMBEDDING in item:
                    item[data.constants.DATA_FRAME_SEQ_LENGTH] = item[data.constants.DATA_FRAME_EMBEDDING].shape[0]
                if data.constants.DATA_AUDIO_EMBEDDING in item:
                    item[data.constants.DATA_AUDIO_SEQ_LENGTH] = item[data.constants.DATA_AUDIO_EMBEDDING].shape[0]
                if data.constants.DATA_PRIMARY_FACE_HOG in item:
                    item[data.constants.DATA_FACE_FEATURES_SEQ_LENGTH] = np.array(item[data.constants.DATA_PRIMARY_FACE_HOG]).shape[0]
                elif data.constants.DATA_PRIMARY_FACE_AUS in item:
                    item[data.constants.DATA_FACE_FEATURES_SEQ_LENGTH] = item[data.constants.DATA_PRIMARY_FACE_AUS].shape[0]
                elif data.constants.DATA_PRIMARY_FACE_LANDMARKS in item:
                    item[data.constants.DATA_FACE_FEATURES_SEQ_LENGTH] = item[data.constants.DATA_PRIMARY_FACE_LANDMARKS].shape[0]
                                
                item[data.constants.DATA_SPEAKER_MASK] = item[data.constants.DATA_SPEAKER]
                item[data.constants.DATA_UTTERANCE_MASK] = [1]

                if data.constants.DATA_PRIMARY_FACE_LANDMARK_GRAPH in item:
                    item[data.constants.DATA_PRIMARY_FACE_LANDMARK_GRAPH] = item[data.constants.DATA_PRIMARY_FACE_LANDMARK_GRAPH].transpose(1, 0, 2, 3)
                if data.constants.DATA_FACE_MOCAP_GRAPH in item:
                    item[data.constants.DATA_FACE_MOCAP_GRAPH] = item[data.constants.DATA_FACE_MOCAP_GRAPH].transpose(1, 0, 2, 3)

            dialog_data.sort(key=lambda x: int(x[data.constants.IEMOCAP_UTTERANCE_INDEX]))

            grouped_data.append(dialog_data)

            if len(dialog_data) > self.max_utterance_length:
                self.max_utterance_length = len(dialog_data)

        for d, dialog in enumerate(grouped_data):
            for u, utterance in enumerate(dialog):
                utterance[data.constants.DATA_INDEX] = d * self.max_utterance_length + u

        return grouped_data
