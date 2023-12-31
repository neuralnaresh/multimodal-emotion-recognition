import os
import subprocess
import struct
import shutil
import typing

import numpy as np

from dlpipeline.data.transform import Transformer

import data.constants
import data.utils

from data.transform.config.extract_openface import ExtractOpenFaceConfig, OpenFaceExtractionType

_openface_exec_map = {
    OpenFaceExtractionType.SINGLE: 'FeatureExtraction',
    OpenFaceExtractionType.MULTI: 'FaceLandmarkVidMulti'
}

class ExtractOpenFace(Transformer):
    def _openface(self, input_video: str, output_dir: str, extraction_type: OpenFaceExtractionType):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        result = subprocess.run([_openface_exec_map[extraction_type], '-f', f'{input_video}', '-out_dir', f'{output_dir}', '-hogalign'], stdout=subprocess.DEVNULL)
        result.check_returncode()

    def _face_hog(self, filename: str):
        batch_size = 5000

        all_feature_vectors = []
        with open(filename, "rb") as f:
            num_cols, = struct.unpack("i", f.read(4))
            num_rows, = struct.unpack("i", f.read(4))
            num_channels, = struct.unpack("i", f.read(4))

            # The first four bytes encode a boolean value whether the frame is valid
            num_features = 1 + num_rows * num_cols * num_channels
            feature_vector = struct.unpack("{}f".format(num_features), f.read(num_features * 4))
            feature_vector = np.array(feature_vector).reshape((1, num_features))
            all_feature_vectors.append(feature_vector)

            # Every frame contains a header of four float values: num_cols, num_rows, num_channels, is_valid
            num_floats_per_feature_vector = 4 + num_rows * num_cols * num_channels
            # Read in batches of given batch_size
            num_floats_to_read = num_floats_per_feature_vector * batch_size
            # Multiply by 4 because of float32
            num_bytes_to_read = num_floats_to_read * 4

            while True:
                bytes = f.read(num_bytes_to_read)
                # For comparison how many bytes were actually read
                num_bytes_read = len(bytes)
                assert num_bytes_read % 4 == 0, "Number of bytes read does not match with float size"
                num_floats_read = num_bytes_read // 4
                assert num_floats_read % num_floats_per_feature_vector == 0, "Number of bytes read does not match with feature vector size"
                num_feature_vectors_read = num_floats_read // num_floats_per_feature_vector

                feature_vectors = struct.unpack("{}f".format(num_floats_read), bytes)
                # Convert to array
                feature_vectors = np.array(feature_vectors).reshape((num_feature_vectors_read, num_floats_per_feature_vector))
                # Discard the first three values in each row (num_cols, num_rows, num_channels)
                feature_vectors = feature_vectors[:, 3:]
                # Append to list of all feature vectors that have been read so far
                all_feature_vectors.append(feature_vectors)

                if num_bytes_read < num_bytes_to_read:
                    break

            # Concatenate batches
            all_feature_vectors = np.concatenate(all_feature_vectors, axis=0)

            # Split into is-valid and feature vectors
            is_valid = all_feature_vectors[:, 0]
            feature_vectors = all_feature_vectors[:, 1:]

            valid_features = []

            for idx in range(is_valid.shape[0]):
                if is_valid[idx] == 1:
                    valid_features.append(feature_vectors[idx])

            if len(valid_features) == 0:
                return None

            return np.stack(valid_features)

    def metadata(self, meta: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        meta[data.constants.META_FACE_HOG_DIM] = 4464
        return meta

    def transform_single(self, input_filename: str, output_base_dir: str):
        data_sample = data.utils.read_pickle(input_filename)

        split = data.utils.split_from_filename(input_filename)
        video_id = data_sample[data.constants.DATA_ID]
        output_dir = f'{output_base_dir}/{split}/openface/{video_id}'

        self._openface(data_sample[data.constants.DATA_VIDEO_PATH], output_dir, ExtractOpenFaceConfig().extraction_type)
        data_sample[data.constants.DATA_OPENFACE_DIR] = output_dir

        hog = self._face_hog(f'{output_dir}/{video_id}.hog')

        if hog is None:
            return None

        data_sample[data.constants.DATA_PRIMARY_FACE_HOG] = hog

        return data_sample

    def write_single(self, processed, input_filename: str, output_base_dir: str) -> None:
        data.utils.write_transformer_pickle(processed, input_filename, output_base_dir)