from cmath import isnan
import os
import pickle
import typing

import numpy as np

def one_hot_encode(label: int, classes: int) -> np.ndarray:
    return np.eye(classes)[label].squeeze()

def split_from_filename(filename: str) -> str:
    return os.path.dirname(filename.replace('\\', '/')).split('/')[-1]

def transformer_output_path(
        input_filename: str, output_base_dir: str, extension: str = ""
    ) -> str:
        input_filename = input_filename.replace("\\", "/")
        output_filename = (
            f'{os.path.basename(input_filename).split(".")[0]}.{extension}'
        )
        return f'{output_base_dir}/{os.path.dirname(input_filename).split("/")[-1]}/{output_filename}'

def read_pickle(filename: str) -> typing.Any:
    with open(filename, 'rb') as f:
        return pickle.load(f)

def write_pickle(data: typing.Any, filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def write_transformer_pickle(data: typing.Any, input_filename: str, output_base_dir: str):
    if data is not None:
        write_pickle(data, transformer_output_path(input_filename, output_base_dir, 'pkl'))

def edge_list_to_adjacency_matrix(edges: typing.List[typing.List[int]], vertice_count: int) -> np.ndarray:
    adj_matrix = np.zeros((vertice_count, vertice_count), dtype=np.float32)
    
    for edge in edges:
        adj_matrix[edge[0]][edge[1]] = 1
    
    return adj_matrix

def k_adjacency(
        adjacency_matrix: np.ndarray,
        k: int,
        with_self: bool = False,
        self_factor: int = 1,
    ) -> np.ndarray:
        vertices = adjacency_matrix.shape[0]
        identity = np.eye(vertices, dtype=adjacency_matrix.dtype)

        if k == 0:
            return identity

        k_adjacency = np.minimum(
            np.linalg.matrix_power(adjacency_matrix + identity, k), 1) \
                - np.minimum(np.linalg.matrix_power(adjacency_matrix + identity, k - 1), 1)

        if with_self:
            k_adjacency = k_adjacency + identity * self_factor

        if np.isnan(k_adjacency).any():
            import pdb
            pdb.set_trace()

        return k_adjacency

def normalize_adjacency(adjacency_matrix: np.ndarray) -> np.ndarray:
        node_degrees = adjacency_matrix.sum(-1)
        degs_inv_sqrt = np.power(node_degrees, -0.5)
        norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt

        return (norm_degs_matrix @ adjacency_matrix @ norm_degs_matrix).astype(np.float32)

def normalize(data: np.ndarray, unitary = False) -> np.ndarray:
    data = data / np.max(data)

    if unitary:
        data = data - 0.5

    return data

def normalize_data_dict(data: typing.List[typing.Dict[str, typing.Any]], keys: typing.Union[str, typing.List[str]], whole: typing.Dict[str, bool] = {}) -> typing.List[typing.Dict[str, typing.Any]]:
    if not len(data):
        return data

    if isinstance(keys, str):
        keys = [keys]

    for k in keys:
        data_max = np.zeros(data[0][k].shape)

        for d in data:
            data_max = np.maximum(data_max, d[k])
        
        for d in data:
            d[k] = d[k] / data_max if (k in whole and not whole[k]) else d[k] / data_max.max()

    return data
