import typing

import numpy as np

from dlpipeline.data.transform import Transformer

import data.constants
import data.utils

from data.transform.config.landmarks_to_graph import LandmarksToGraphConfig

class LandmarksToGraph(Transformer):
    def _sample_frame_indices(self, clip_len, seg_len):
        return np.linspace(0, seg_len - 1, clip_len).astype(np.int64)

    def _extract_frames(self, landmarks: typing.List) -> np.ndarray:
        if (len(landmarks) < LandmarksToGraphConfig().frames):
            landmarks = np.array(landmarks + [landmarks[-1]] * (len(landmarks) - LandmarksToGraphConfig().frames))
        else:
            indices = self._sample_frame_indices(LandmarksToGraphConfig().frames, len(landmarks))
            landmarks = np.array(landmarks)[indices]

        return landmarks

    def _import_graph_alogs(self):
        import pyximport
        pyximport.install(setup_args={"include_dirs": np.get_include()})

        from transformers.models.graphormer import algos_graphormer

        return algos_graphormer

    def _convert_to_single_emb(self, x, offset: int = 512):
        feature_num = x.shape[1] if len(x.shape) > 1 else 1
        feature_offset = 1 + np.arange(0, feature_num * offset, offset, dtype=np.int32)
        x = x + feature_offset
        return x

    def _preprocess(self, item, keep_features=True):
        algos_graphormer = self._import_graph_alogs()

        if keep_features and "edge_attr" in item.keys():  # edge_attr
            edge_attr = np.asarray(item["edge_attr"], dtype=np.int32)
        else:
            edge_attr = np.ones((len(item["edge_index"][0]), 1), dtype=np.int32)  # same embedding for all

        if keep_features and "node_feat" in item.keys():  # input_nodes
            node_feature = np.asarray(item["node_feat"], dtype=np.int32)
        else:
            node_feature = np.ones((item["num_nodes"], 1), dtype=np.int32)  # same embedding for all

        edge_index = np.asarray(item["edge_index"], dtype=np.int32)

        input_nodes = self._convert_to_single_emb(node_feature) + 1
        num_nodes = item["num_nodes"]

        if len(edge_attr.shape) == 1:
            edge_attr = edge_attr[:, None]
        attn_edge_type = np.zeros([num_nodes, num_nodes, edge_attr.shape[-1]], dtype=np.int32)
        attn_edge_type[edge_index[0], edge_index[1]] = self._convert_to_single_emb(edge_attr) + 1

        # node adj matrix [num_nodes, num_nodes] bool
        adj = np.zeros([num_nodes, num_nodes], dtype=bool)
        adj[edge_index[0], edge_index[1]] = True

        shortest_path_result, path = algos_graphormer.floyd_warshall(adj)
        max_dist = np.amax(shortest_path_result)

        input_edges = algos_graphormer.gen_edge_input(max_dist, path, attn_edge_type)
        attn_bias = np.zeros([num_nodes + 1, num_nodes + 1], dtype=np.single)  # with graph token

        # combine
        item["input_nodes"] = input_nodes + 1  # we shift all indices by one for padding
        item["attn_bias"] = attn_bias
        item["attn_edge_type"] = attn_edge_type
        item["spatial_pos"] = shortest_path_result.astype(np.int64) + 1  # we shift all indices by one for padding
        item["in_degree"] = np.sum(adj, axis=1).reshape(-1) + 1  # we shift all indices by one for padding
        item["out_degree"] = item["in_degree"]  # for undirected graph
        item["input_edges"] = input_edges + 1  # we shift all indices by one for padding

        return item

    def _convert_to_graph(self, landmarks: typing.List):
        landmarks = self._extract_frames(landmarks)

        if not LandmarksToGraphConfig().edges_first:
            frames = landmarks.shape[0]
            vertices = landmarks.shape[1]
            channels = landmarks.shape[2]
            persons = 1

            graph = np.zeros((channels, frames, vertices, persons))

            for frame_index, frame in enumerate(landmarks):
                for channel in range(channels):
                    graph[channel, frame_index, :, 0] = frame[:, channel]
            
            for channel in range(channels):
                graph[channel] = data.utils.normalize(graph[channel], True)

            return graph
        else:
            frames = landmarks.shape[0]
            vertices = landmarks.shape[1]
            channels = landmarks.shape[2]

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

            num_nodes = frames * vertices
            edges = [[], []]
            node_features = []
            edge_attributes = []

            for frame in range(frames):
                for edge in inward_connections:
                    edges[0].append(edge[0] * (frame + 1))
                    edges[1].append(edge[1] * (frame + 1))
                node_features.extend(landmarks[frame].tolist())
                edge_attributes.extend([1] * len(inward_connections))

                # add temporal connections
                if frame > 0:
                    for vertex in range(vertices):
                        edges[0].append(vertex + frame * vertices)
                        edges[1].append(vertex + (frame - 1) * vertices)
                    edge_attributes.extend([2] * vertices)

            item = {
                "num_nodes": num_nodes,
                "edge_index": edges,
                "node_feat": node_features,
                "edge_attr": edge_attributes
            }

            return self._preprocess(item)

    def  transform_single(self, input_filename: str, output_base_dir: str):
        config = LandmarksToGraphConfig()

        data_sample = data.utils.read_pickle(input_filename)

        if config.convert_all_faces:
            raise NotImplementedError
        else:
            if len(np.array(data_sample[data.constants.DATA_PRIMARY_FACE_LANDMARKS]).shape) < 3:
                return None

            data_sample[data.constants.DATA_PRIMARY_FACE_LANDMARK_GRAPH] = self._convert_to_graph(data_sample[data.constants.DATA_PRIMARY_FACE_LANDMARKS])

        if config.convert_mocap:
            data_sample[data.constants.DATA_FACE_MOCAP_GRAPH] = self._convert_to_graph(data_sample[data.constants.DATA_FACE_MOCAP])

        data.utils.write_transformer_pickle(data_sample, input_filename, output_base_dir)
        return None

    def write_single(self, processed, input_filename: str, output_base_dir: str) -> None:
        return

        

        
