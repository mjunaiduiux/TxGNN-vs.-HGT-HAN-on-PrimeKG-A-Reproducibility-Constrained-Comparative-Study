import torch.nn as nn
from torch_geometric.nn import HANConv


class HANBaseline(nn.Module):
    """HAN encoder (attention over metapaths) + dot-product decoder. Same hidden dim as HGT."""

    def __init__(self, node_types, metadata, type_counts, hidden=64, num_heads=4, num_layers=2):
        super().__init__()
        self.embed = nn.ModuleDict({t: nn.Embedding(type_counts[t], hidden) for t in node_types})
        self.convs = nn.ModuleList([
            HANConv(hidden, hidden, metadata, heads=num_heads) for _ in range(num_layers)
        ])

    def forward(self, edge_index_dict):
        x_dict = {t: emb.weight for t, emb in self.embed.items()}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: v.relu() for k, v in x_dict.items()}
        return x_dict

    def decode(self, drug_emb, disease_emb, drug_idx, disease_idx):
        return (drug_emb[drug_idx] * disease_emb[disease_idx]).sum(dim=-1)
