"""
gacl/hgavit.py

HGAViT -- Hierarchical Geometry-Agnostic Vision Transformer (Section 7.3).

f_theta : I -> z in R^d

Implements, verbatim from the paper:

    x_n^(0) = x_n + MLP_geo(g)

    Attn_h(X^(l)) = softmax(Q_h K_h^T / sqrt(d_h)) V_h
    X^(l+1) = X^(l) + MLP(Concat_h(Attn_h) W^O)

    hierarchical pooling Pi applied every L/S blocks -> {z^(1), ..., z^(S)}
    z = mean_s(z^(s))

DATA CAVEAT
-----------
The paper's HGAViT patchifies a real RGB image (H x W x 3) via a standard
ViT patch embedding (Dosovitskiy et al. 2021). The dataset shipped with this
repository (GACL_Data.xlsx) contains no images -- only 32-dimensional
pre-computed "embedding_*" columns. There is no way to recover genuine
patch-level spatial structure from those 32 numbers, so `PatchEmbed` below
supports two modes:

  - "image": the real mode described in the paper (I -> N patches of PxP).
    Provided for completeness / for use once real imagery exists
    (see paper Section 10).
  - "vector": a fallback used only so this repository's train.py can be run
    end-to-end against GACL_Data.xlsx. It reshapes each 32-dim embedding
    into (num_patches, patch_dim) pseudo-patches. This is a bookkeeping
    convenience, NOT a claim that those pseudo-patches carry genuine
    spatial/geometric meaning.
"""

import math
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Patch embedding. Supports 'image' (ViT-style) and 'vector' (fallback) modes."""

    def __init__(self, mode="vector", patch_dim=4, num_patches=8, d_model=64,
                 img_size=224, patch_size=16, in_chans=3):
        super().__init__()
        self.mode = mode
        if mode == "image":
            assert img_size % patch_size == 0
            self.num_patches = (img_size // patch_size) ** 2
            self.proj = nn.Conv2d(in_chans, d_model, kernel_size=patch_size, stride=patch_size)
        elif mode == "vector":
            self.num_patches = num_patches
            self.proj = nn.Linear(patch_dim, d_model)
        else:
            raise ValueError(f"Unknown PatchEmbed mode: {mode}")

    def forward(self, x):
        if self.mode == "image":
            # x: (B, 3, H, W) -> (B, N, d_model)
            x = self.proj(x)                      # (B, d, H/P, W/P)
            x = x.flatten(2).transpose(1, 2)       # (B, N, d)
            return x
        else:
            # x: (B, num_patches, patch_dim) -> (B, num_patches, d_model)
            return self.proj(x)


class GeoConditioning(nn.Module):
    """MLP_geo(g): projects acquisition-geometry descriptor into patch-embedding space."""

    def __init__(self, geo_dim=5, d_model=64, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(geo_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, g):
        # g: (B, geo_dim) -> (B, d_model) -> broadcast-added to every patch token
        return self.mlp(g)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        H, d_h = self.num_heads, self.d_head

        Q = self.q_proj(x).view(B, N, H, d_h).transpose(1, 2)  # (B,H,N,d_h)
        K = self.k_proj(x).view(B, N, H, d_h).transpose(1, 2)
        V = self.v_proj(x).view(B, N, H, d_h).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_h)  # (B,H,N,N)
        attn = attn_scores.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ V                                # (B,H,N,d_h)
        out = out.transpose(1, 2).reshape(B, N, D)     # (B,N,D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """X^(l+1) = X^(l) + MLP(Concat_h(Attn_h) W^O), with pre-norm + residual MLP."""

    def __init__(self, d_model, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HGAViT(nn.Module):
    """
    Hierarchical Geometry-Agnostic Vision Transformer.

    forward(x, g) -> (z, multi_scale_list)
        x : (B, num_patches, patch_dim) in 'vector' mode, or (B,3,H,W) in 'image' mode
        g : (B, geo_dim) acquisition-geometry descriptor
        z : (B, d_model) pooled global descriptor, mean over scales
        multi_scale_list : list of S per-scale pooled tensors (B, d_model)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(
            mode="vector",
            patch_dim=cfg.patch_dim,
            num_patches=cfg.num_patches,
            d_model=cfg.d_model,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches, cfg.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.geo_cond = GeoConditioning(cfg.geo_dim, cfg.d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])

        # hierarchical pooling every L/S blocks
        self.pool_every = max(1, cfg.num_layers // cfg.num_scales)

    def forward(self, x, g):
        # x_n = E_vec(I_n) + e_n^pos
        tokens = self.patch_embed(x) + self.pos_embed

        # x_n^(0) = x_n + MLP_geo(g)   (broadcast geometry term over all patches)
        geo_term = self.geo_cond(g).unsqueeze(1)  # (B,1,d)
        tokens = tokens + geo_term

        multi_scale = []
        for i, block in enumerate(self.blocks, start=1):
            tokens = block(tokens)
            if i % self.pool_every == 0 and len(multi_scale) < self.cfg.num_scales:
                # Pi: hierarchical pooling operator (mean pool over patch dim)
                multi_scale.append(tokens.mean(dim=1))

        if not multi_scale:
            multi_scale = [tokens.mean(dim=1)]

        z = torch.stack(multi_scale, dim=0).mean(dim=0)  # z = mean_s(z^(s))
        return z, multi_scale
