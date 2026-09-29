import torch
import torch.nn as nn

from .prototype_pooling import cosine_similarity_map, masked_average_pooling


class SSPRefinement(nn.Module):
    def __init__(self, stage_channels, temperature=10.0, mode="soft", top_ratio=0.15):
        super().__init__()
        if mode not in {"soft", "topk"}:
            raise ValueError("ssp mode must be soft or topk")
        self.stage_channels = list(stage_channels)
        self.temperature = float(temperature)
        self.mode = mode
        self.top_ratio = float(top_ratio)
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(ch * 4, ch),
                    nn.ReLU(inplace=True),
                    nn.Linear(ch, ch),
                    nn.Sigmoid(),
                )
                for ch in self.stage_channels
            ]
        )

    def _weights(self, similarity):
        if self.mode == "soft":
            return torch.sigmoid(similarity)
        b, _, h, w = similarity.shape
        flat = similarity.flatten(2)
        k = max(1, int(flat.shape[-1] * self.top_ratio))
        threshold = flat.topk(k, dim=-1).values[..., -1:]
        return (flat >= threshold).float().reshape(b, 1, h, w)

    def forward(self, support_prototypes, query_features):
        refined, initial_maps, weights = [], [], []
        for idx, (proto, feat) in enumerate(zip(support_prototypes, query_features)):
            sim = cosine_similarity_map(feat, proto, temperature=self.temperature)
            weight = self._weights(sim)
            query_proto = masked_average_pooling(feat, weight)
            gate_input = torch.cat([proto, query_proto, (proto - query_proto).abs(), proto * query_proto], dim=1)
            gate = self.gates[idx](gate_input)
            refined_proto = gate * proto + (1.0 - gate) * query_proto
            refined.append(torch.nan_to_num(refined_proto))
            initial_maps.append(torch.nan_to_num(sim))
            weights.append(torch.nan_to_num(weight))
        return {
            "refined_prototypes": refined,
            "initial_similarity_maps": initial_maps,
            "ssp_weights": weights,
        }

