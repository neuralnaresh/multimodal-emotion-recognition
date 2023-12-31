import os
import enum
import typing
import random
import numbers

import cv2
import skimage.transform
import vidaug.augmentors

import numpy as np

from dlpipeline.data.transform import Transformer

import data.constants
import data.utils

from data.transform.config.augment_frames import AugmentFramesConfig

# FIX: vidaug RandomRotate doesn't preserve range, and outputs a black image

class RandomRotate(object):
    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [skimage.transform.rotate(img, angle, preserve_range=True) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return rotated

# Intensity implements both salt and pepper from vidaug and allows entire pixel intensity to go to 0/255 instead of single channel like vidaug

class NoiseType(enum.Enum):
    SALT = 'salt'
    PEPPER = 'pepper'

class Noise(object):
    """
    Augmenter that sets a certain fraction of pixel intensities to 0, hence
    they become black.
    Args:
        ratio (int): Determines number of black pixels on each frame of video.
        Smaller the ratio, higher the number of black pixels.
    """
    def __init__(self, type: NoiseType, ratio=100):
        self.type = type
        self.ratio = ratio

    def __call__(self, clip):
        data_final = []
        
        for i in range(len(clip)):
            img = clip[i].astype(np.float)
            img_shape = img.shape
            noise = np.stack([np.random.randint(self.ratio, size=img_shape[:-1])] * img_shape[-1], axis=-1)
            img = np.where(noise == 0, 0 if self.type == NoiseType.PEPPER else 255, img)
            data_final.append(img.astype(np.uint8))

        return data_final

class AugmentFrames(Transformer):
    def __init__(self) -> None:
        super().__init__()

        self.augment = vidaug.augmentors.Sequential(
            [
                RandomRotate(degrees=15),
                Noise(NoiseType.PEPPER, 98),
                Noise(NoiseType.SALT, 98),
                vidaug.augmentors.Salt(98),
                vidaug.augmentors.Pepper(98),
                vidaug.augmentors.Sometimes(0.5, vidaug.augmentors.HorizontalFlip()),
            ]
        )

    def metadata(self, meta: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        meta[data.constants.META_N_TRAIN] = meta[data.constants.META_N_TRAIN] * (AugmentFramesConfig().augmentations + 1)
        return meta

    def transform_single(
        self, input_filename: str, output_base_dir: str
    ) -> typing.List[typing.Dict[str, typing.Any]]:
        data_sample = data.utils.read_pickle(input_filename)

        split = data.utils.split_from_filename(input_filename)

        if not split == "train":
            return [data_sample]

        frames_dir = data_sample[data.constants.DATA_FRAMES_DIR]
        frames = [
            cv2.imread(f"{frames_dir}/{frame_path}")
            for frame_path in sorted(os.listdir(frames_dir))
        ]

        results = [data_sample]

        for i in range(AugmentFramesConfig().augmentations):
            augmented_frames = self.augment(frames)

            augmentated_frames_path = f"{frames_dir}_augmented_{i}"
            os.makedirs(augmentated_frames_path, exist_ok=True)

            augmented_sample = data_sample.copy()
            augmented_sample[data.constants.DATA_FRAMES_DIR] = augmentated_frames_path

            for index, frame in enumerate(augmented_frames):
                cv2.imwrite(f"{augmentated_frames_path}/{index:0>4d}.png", frame)

            results.append(augmented_sample)

        return results

    def write_single(
        self,
        processed: typing.List[typing.Dict[str, typing.Any]],
        input_filename: str,
        output_base_dir: str,
    ) -> None:
        for i, sample in enumerate(processed):
            data.utils.write_transformer_pickle(
                sample, f'{input_filename.split(".")[0]}_aug_{i}', output_base_dir
            )
