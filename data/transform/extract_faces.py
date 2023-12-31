import os
import glob

import cv2
import facenet_pytorch

import numpy as np

from dlpipeline.data.transform import Transformer
from dlpipeline.commands.experiment_config import ExperimentConfig

import data.constants
import data.utils

from data.transform.config.extract_faces import ExtractFacesConfig

class ExtractFaces(Transformer):
    def __init__(self) -> None:
        super().__init__()

        self.mtcnn = facenet_pytorch.MTCNN(keep_all=True, device=ExperimentConfig().device)

    def _extract_faces_and_bg(self, frames_dir: str, faces_output_dir: str, bg_output_dir: str) -> None:
        os.makedirs(faces_output_dir, exist_ok=True)
        os.makedirs(bg_output_dir, exist_ok=True)

        frames = glob.glob(frames_dir + '/*.png')

        for frame_path in frames:
            frame = cv2.imread(frame_path)
            boxes, _ = self.mtcnn.detect(frame)

            buffer = ExtractFacesConfig().buffer_size

            faces = np.zeros(frame.shape, dtype=np.uint8)
            bg = np.array(frame)

            if boxes is not None:
                for box in boxes:
                    y,x,h,w = box
            
                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)

                    faces[x-buffer:w+buffer,y-buffer:h+buffer] = frame[x-buffer:w+buffer,y-buffer:h+buffer]
                    bg[x-buffer:w+buffer,y-buffer:h+buffer] = np.zeros(bg[x-buffer:w+buffer,y-buffer:h+buffer].shape, dtype=np.uint8)

            cv2.imwrite(f'{faces_output_dir}/{os.path.basename(frame_path)}', faces)
            cv2.imwrite(f'{bg_output_dir}/{os.path.basename(frame_path)}', bg)

    def transform_single(self, input_filename: str, output_base_dir: str):
        data_sample = data.utils.read_pickle(input_filename)

        split = data.utils.split_from_filename(input_filename)

        frames_dir = data_sample[data.constants.DATA_FRAMES_DIR]
        faces_output_dir = f'{output_base_dir}/{split}/frames_faces/{data_sample[data.constants.DATA_ID]}'
        bg_output_dir = f'{output_base_dir}/{split}/frames_bg/{data_sample[data.constants.DATA_ID]}'

        if not os.path.exists(faces_output_dir) and not os.path.exists(bg_output_dir):
            self._extract_faces_and_bg(frames_dir, faces_output_dir, bg_output_dir)

        data_sample[data.constants.DATA_FRAME_FACES_DIR] = faces_output_dir
        data_sample[data.constants.DATA_FRAME_BG_DIR] = bg_output_dir

        return data_sample

    def write_single(self, processed, input_filename: str, output_base_dir: str) -> None:
        data.utils.write_transformer_pickle(processed, input_filename, output_base_dir)
