import os
import glob
import pickle
import typing

import cv2
import torch
import torchvision

import numpy as np

from dlpipeline.data.transform import Transformer
from dlpipeline.commands.experiment_config import ExperimentConfig

import data.constants
import data.utils

from data.transform.config.embed_frames import EmbedFramesConfig, EmbedFramesSource

class EmbedVideo(Transformer):
    def __init__(self) -> None:
        super().__init__()

        self.device = ExperimentConfig().device
        self.model = self._make_model()

    def _make_model(self):
        model = torchvision.models.resnet101(pretrained=True)
        modules = list(model.children())[:-1]
        model = torch.nn.Sequential(*modules)
        for p in model.parameters():
            p.requires_grad = False

        model.to(self.device)
        model.eval()

        return model

    def _embed_frames(self, frames_dir: str, batch_size: int):
        frame_paths = sorted(glob.glob(frames_dir + '/*.png'))
        frames = []

        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, (224, 224))
            frame = frame.transpose((2, 0, 1))

            frames.append(frame)
        
        if len(frames) <= batch_size:
            frames = np.array(frames)
            frames = torch.from_numpy(frames).float()
            frames = frames.to(self.device)

            embeddings = self.model(frames)
            embeddings = embeddings.detach().cpu().numpy()
        else:
            embeddings = []
            for i in range(len(frames) // batch_size):
                frames_to_process = frames[i*batch_size:max((i+1)*batch_size, len(frames))]
                frames_to_process = np.array(frames_to_process)
                frames_to_process = torch.from_numpy(frames_to_process).float()
                frames_to_process = frames_to_process.to(self.device)

                embeddings.append(self.model(frames_to_process).detach().cpu().numpy())
            
            embeddings = np.concatenate(embeddings, axis=0)

        return embeddings.squeeze()

    def _get_frames_dir(self, data_sample: typing.Dict[str, typing.Any], source: EmbedFramesSource) -> str:
        if source == EmbedFramesSource.FRAMES:
            return data_sample[data.constants.DATA_FRAMES_DIR]
        elif source == EmbedFramesSource.FACES:
            return data_sample[data.constants.DATA_FRAME_FACES_DIR]
        elif source == EmbedFramesSource.BG:
            return data_sample[data.constants.DATA_FRAME_BG_DIR]
        
    def metadata(self, meta: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        meta[data.constants.META_FRAME_EMBEDDIG_DIM] = 2048
        return meta

    def transform_single(self, input_filename: str, output_base_dir: str):
        data_sample = data.utils.read_pickle(input_filename)

        try:
            if os.path.exists(Transformer.get_output_path(input_filename, output_base_dir, 'pkl')):
                return pickle.load(open(Transformer.get_output_path(input_filename, output_base_dir, 'pkl'), 'rb'))

            frames_dir = self._get_frames_dir(data_sample, EmbedFramesConfig().source)
            data_sample[data.constants.DATA_FRAME_EMBEDDING] = self._embed_frames(frames_dir, EmbedFramesConfig().batch_size)

            return data_sample
        except:
            return None

    def write_single(self, processed, input_filename: str, output_base_dir: str) -> None:
        data.utils.write_transformer_pickle(processed, input_filename, output_base_dir)
