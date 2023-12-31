import typing

import torch

from transformers import GraphormerModel

from layers.config.graphormerencoder import GraphormerEncoderConfig

class GraphormerEncoder(torch.nn.Module):
    def __init__(self, config: GraphormerEncoderConfig) -> None:
        super().__init__()

        self.graphormer = GraphormerModel.from_pretrained("clefourrier/pcqm4mv2_graphormer_base")

        self.dropout = torch.nn.Dropout(p=0.1)
        self.linear = torch.nn.Linear(self.graphormer.config.hidden_size, config.output_size)

    @classmethod
    def collate(cls, input: typing.List[typing.Dict[str, typing.Any]]) -> typing.Dict[str, torch.Tensor]:
        spatial_pos_max = 20

        preprocessed = [x['primary_face_landmark_graph'] for x in input]

        batch = {}

        max_node_num = max(len(i["input_nodes"]) for i in preprocessed)
        node_feat_size = len(preprocessed[0]["input_nodes"][0])
        edge_feat_size = len(preprocessed[0]["attn_edge_type"][0][0])
        max_dist = max(len(i["input_edges"][0][0]) for i in preprocessed)
        edge_input_size = len(preprocessed[0]["input_edges"][0][0][0])
        batch_size = len(preprocessed)

        batch["attn_bias"] = torch.zeros(batch_size, max_node_num + 1, max_node_num + 1, dtype=torch.float)
        batch["attn_edge_type"] = torch.zeros(batch_size, max_node_num, max_node_num, edge_feat_size, dtype=torch.long)
        batch["spatial_pos"] = torch.zeros(batch_size, max_node_num, max_node_num, dtype=torch.long)
        batch["in_degree"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["input_nodes"] = torch.zeros(batch_size, max_node_num, node_feat_size, dtype=torch.long)
        batch["input_edges"] = torch.zeros(
            batch_size, max_node_num, max_node_num, max_dist, edge_input_size, dtype=torch.long
        )

        for ix, f in enumerate(preprocessed):
            for k in ["attn_bias", "attn_edge_type", "spatial_pos", "in_degree", "input_nodes", "input_edges"]:
                f[k] = torch.tensor(f[k])

            if len(f["attn_bias"][1:, 1:][f["spatial_pos"] >= spatial_pos_max]) > 0:
                f["attn_bias"][1:, 1:][f["spatial_pos"] >= spatial_pos_max] = float("-inf")

            batch["attn_bias"][ix, : f["attn_bias"].shape[0], : f["attn_bias"].shape[1]] = f["attn_bias"]
            batch["attn_edge_type"][ix, : f["attn_edge_type"].shape[0], : f["attn_edge_type"].shape[1], :] = f[
                "attn_edge_type"
            ]
            batch["spatial_pos"][ix, : f["spatial_pos"].shape[0], : f["spatial_pos"].shape[1]] = f["spatial_pos"]
            batch["in_degree"][ix, : f["in_degree"].shape[0]] = f["in_degree"]
            batch["input_nodes"][ix, : f["input_nodes"].shape[0], :] = f["input_nodes"]
            batch["input_edges"][
                ix, : f["input_edges"].shape[0], : f["input_edges"].shape[1], : f["input_edges"].shape[2], :
            ] = f["input_edges"]

        batch["out_degree"] = batch["in_degree"]

        return batch

    def forward(self, graph: typing.Dict[str, torch.Tensor]) -> torch.Tensor:
        graph["attn_bias"] = graph["attn_bias"].to(self.graphormer.device)
        graph["attn_edge_type"] = graph["attn_edge_type"].to(self.graphormer.device)
        graph["spatial_pos"] = graph["spatial_pos"].to(self.graphormer.device)
        graph["in_degree"] = graph["in_degree"].to(self.graphormer.device)
        graph["input_nodes"] = graph["input_nodes"].to(self.graphormer.device)
        graph["input_edges"] = graph["input_edges"].to(self.graphormer.device)
        graph["out_degree"] = graph["out_degree"].to(self.graphormer.device)

        output = self.graphormer(**graph).last_hidden_state

        output = self.dropout(output)
        output = self.linear(output)

        output = output[:, 0, :]

        return torch.relu(output)
