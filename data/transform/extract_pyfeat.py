import os
import glob
import typing

import cv2

import numpy as np

from feat import Detector

from dlpipeline.data.transform import Transformer

from data.transform.config.extract_pyfeat import ExtractPyFeatConfig

import data.constants
import data.utils

class ExtractPyFeat(Transformer):
    def __init__(self) -> None:
        super().__init__()

        config = ExtractPyFeatConfig()

        self.detector = Detector(
            face_model=config.face_model,
            landmark_model=config.landmark_model,
            au_model=config.au_model,
            emotion_model=config.emotion_model,
            facepose_model=config.facepose_model
        )

        self._connections = self._get_landmark_connections()

    def _get_landmark_connections(self) -> np.ndarray:
        # [00:17] - Jaw
        # [17:22] - left eyebrow
        # [22:27] - right eyebrow
        # [27:36] - nose
        # [36:42] - left eye
        # [42:48] - right eye
        # [48:60] - outer mouth
        # [60:68] - inner mouth

        connections = [(i, i+1) for i in range(67)]
        connection_pieces = [0, 17, 22, 27, 36, 42, 48, 60, 68]

        inward_connections = [connection for i in range(len(connection_pieces)-1) for connection in connections[connection_pieces[i]:connection_pieces[i+1]-1]]
        outward_connections = [(j, i) for i, j in inward_connections]
        self_connections = [(i, i) for i in range(68)]

        return data.utils.edge_list_to_adjacency_matrix(inward_connections + outward_connections + self_connections, 68)

    def metadata(self, meta: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        meta[data.constants.META_FACE_HOG_DIM] = 5408
        meta[data.constants.META_FACE_LANDMARK_DIM] = 68
        meta[data.constants.META_FACE_LANDMARK_COORD_DIM] = 2
        meta[data.constants.META_FACE_LANDMARK_CONNECTIONS] = self._connections.tolist()
        meta[data.constants.META_FACE_AU_DIM] = 12

        return meta

    def transform_single(self, input_filename: str, output_base_dir: str):
        output_file = Transformer.get_output_path(input_filename, output_base_dir, 'pkl')
        if os.path.exists(output_file):
            return data.utils.read_pickle(output_file)

        data_sample = data.utils.read_pickle(input_filename)

        frames_dir = data_sample[data.constants.DATA_FRAMES_DIR]

        images = list(sorted(glob.glob(f'{frames_dir}/*.png')))
        
        primary_face_hog = []
        primary_face_landmarks = []
        primary_face_aus = []

        secondary_faces_hog = []
        secondary_faces_landmarks = []
        secondary_faces_aus = []

        for image in images:
            img = cv2.imread(image, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            frames = np.expand_dims(img, 0)
            
            detected_faces = self.detector.detect_faces(frames)

            if detected_faces[0] is None:
                if len(primary_face_hog) > 0:
                    primary_face_hog.append(primary_face_hog[-1])
                    primary_face_landmarks.append(primary_face_landmarks[-1])
                    primary_face_aus.append(primary_face_aus[-1])

                    secondary_faces_hog.append(np.zeros((0, 5048)))
                    secondary_faces_landmarks.append(np.zeros((0, 68, 2)))
                    secondary_faces_aus.append(np.zeros((0, 12)))
                
                continue

            landmarks = self.detector.detect_landmarks(frame=frames, detected_faces=detected_faces)
            hog_landmarks = landmarks
            hog, hog_landmarks = self.detector._batch_hog(frames=frames, detected_faces=detected_faces, landmarks=hog_landmarks)
            au_occur = self.detector.detect_aus(frame=frames, landmarks=landmarks)

            primary_face_index = 0
            largest_face_area = 0

            for i, face in enumerate(detected_faces[0]):
                face_area = (face[2] - face[0]) * (face[3] - face[1])

                if face_area > largest_face_area:
                    largest_face_area = face_area
                    primary_face_index = i
            
            primary_face_hog.append(hog[primary_face_index])
            primary_face_landmarks.append(landmarks[0][primary_face_index])
            primary_face_aus.append(au_occur[primary_face_index])

            secondary_faces_hog.append(np.zeros((0, 5048)) if len(hog) == 1 else np.concatenate([hog[i] for i in range(len(hog)) if i != primary_face_index]))
            secondary_faces_landmarks.append(np.zeros((0, 68, 2)) if len(landmarks[0]) == 1 else np.concatenate([landmarks[0][i] for i in range(len(landmarks[0])) if i != primary_face_index]))
            secondary_faces_aus.append(np.zeros((0, 12)) if len(au_occur) == 1 else np.concatenate([au_occur[i] for i in range(len(au_occur)) if i != primary_face_index]))

        if len(primary_face_hog) == 0:
            return None

        data_sample[data.constants.DATA_FACE_FEATURES_SEQ_LENGTH] = len(images)
        
        data_sample[data.constants.DATA_PRIMARY_FACE_HOG] = primary_face_hog
        data_sample[data.constants.DATA_PRIMARY_FACE_LANDMARKS] = primary_face_landmarks
        data_sample[data.constants.DATA_PRIMARY_FACE_AUS] = primary_face_aus

        data_sample[data.constants.DATA_SECONDARY_FACES_HOG] = secondary_faces_hog
        data_sample[data.constants.DATA_SECONDARY_FACES_LANDMARKS] = secondary_faces_landmarks
        data_sample[data.constants.DATA_SECONDARY_FACES_AUS] = secondary_faces_aus

        self.write_single(data_sample, input_filename, output_base_dir)

        return data_sample

    def write_single(self, processed, input_filename: str, output_base_dir: str) -> None:
        data.utils.write_transformer_pickle(processed, input_filename, output_base_dir)