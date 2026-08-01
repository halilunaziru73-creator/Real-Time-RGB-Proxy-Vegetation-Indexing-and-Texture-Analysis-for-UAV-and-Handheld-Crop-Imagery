"""
gacl/dhgnn.py

DHGNN -- Dynamic Hypergraph Neural Network (Section 7.6).

H = (V, E), incidence matrix H in {0,1}^(|V| x |E|), constructed dynamically
per batch from k-NN graphs over each of the crop, pathology, and
geometry-descriptor spaces (Jiang et al. 2019 dynamic-hypergraph construction).

Z^(l+1) = sigma( D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} Z^(l) Theta^(l) )
"""

import torch
import torch.nn as nn


def _knn_incidence(features, k):
    """
    Build a k-NN hyperedge incidence block for one modality's feature space.
    Each node is the centroid of a hyperedge connecting itself + its k nearest
    neighbours (standard k-NN hypergraph construction).

    features: (N, F) tensor for one modality (e.g. geometry descriptor space)
    returns:  (N, N) incidence block, columns = hyperedges (one per node)
    """
    N = features.shape[0]
    k = min(k, N - 1) if N > 1 else 0
    if k <= 0:
        return torch.eye(N, device=features.device)

    dist = torch.cdist(features, features, p=2)          # (N,N)
    # exclude self before top-k, then re-add self manually
    dist.fill_diagonal_(float("inf"))
    _, knn_idx = torch.topk(dist, k=k, largest=False, dim=-1)  # (N,k)

    H = torch.zeros(N, N, device=features.device)
    node_idx = torch.arange(N, device=features.device).unsqueeze(1).expand(-1, k)
    H[node_idx.reshape(-1), knn_idx.reshape(-1)] = 1.0
    H[torch.arange(N), torch.arange(N)] = 1.0  # hyperedge includes its own centroid
    return H


def build_dynamic_incidence(crop_ids, pathology_logits_or_labels, geo_features, k=8):
    """
    Constructs H by concatenating k-NN hyperedge blocks over three modalities:
    crop identity (one-hot), pathology (label or soft label), and acquisition
    geometry (continuous). Concatenation along the hyperedge axis follows
    Jiang et al.'s multi-modal dynamic hypergraph construction.

    crop_ids: (N,) long
    pathology_logits_or_labels: (N,) long, or (N,C) float
    geo_features: (N, G) float
    returns: H (N, E) with E = 3N (one hyperedge set per modality)
    """
    device = geo_features.device
    N = geo_features.shape[0]

    crop_onehot = torch.nn.functional.one_hot(crop_ids).float().to(device)
    if pathology_logits_or_labels.dim() == 1:
        path_feat = torch.nn.functional.one_hot(pathology_logits_or_labels).float().to(device)
    else:
        path_feat = pathology_logits_or_labels.to(device)

    H_crop = _knn_incidence(crop_onehot, k)
    H_path = _knn_incidence(path_feat, k)
    H_geo = _knn_incidence(geo_features, k)

    H = torch.cat([H_crop, H_path, H_geo], dim=1)  # (N, 3N)
    return H


class HypergraphConv(nn.Module):
    """One layer of Z^(l+1) = sigma(D_v^-1/2 H W_e D_e^-1 H^T D_v^-1/2 Z^(l) Theta^(l))."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.theta = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, Z, H):
        # H: (N, E)
        device = Z.device
        d_v = H.sum(dim=1).clamp(min=1e-6)   # (N,) node degree
        d_e = H.sum(dim=0).clamp(min=1e-6)   # (E,) hyperedge degree

        Dv_inv_sqrt = torch.diag(d_v.pow(-0.5))
        De_inv = torch.diag(d_e.pow(-1.0))
        W_e = torch.eye(H.shape[1], device=device)  # learnable hyperedge weights (identity init)

        ZT = self.theta(Z)  # Z^(l) Theta^(l)

        # D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2}
        M = Dv_inv_sqrt @ H @ W_e @ De_inv @ H.t() @ Dv_inv_sqrt
        out = M @ ZT
        return out


class DHGNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dims = [cfg.d_model] + [cfg.hidden_dim] * (cfg.num_layers - 1) + [cfg.d_model]
        self.layers = nn.ModuleList([
            HypergraphConv(dims[i], dims[i + 1]) for i in range(cfg.num_layers)
        ])
        self.act = nn.GELU()
        # node-classification head for L_hypergraph (Section 7.8)
        self.classifier = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, Z, crop_ids, pathology_labels, geo_features):
        H = build_dynamic_incidence(crop_ids, pathology_labels, geo_features, k=self.cfg.knn_k)
        out = Z
        for i, layer in enumerate(self.layers):
            out = layer(out, H)
            if i < len(self.layers) - 1:
                out = self.act(out)
        logits = self.classifier(out)  # used by L_hypergraph node-classification CE
        return out, logits
