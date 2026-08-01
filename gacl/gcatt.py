"""
gacl/gcatt.py

GCATT -- Geometry-Aware Cross-Attention Transfer Transformer (Section 7.5).

CrossAttn(z^(q), {mu_k}) = sum_k alpha_k * W_v mu_k
    alpha_k = softmax_k'( z^(q)^T W_q^T W_k mu_k' / sqrt(d) )

L_proto = -log [ exp(-||z_i - mu_{p_i}||^2) / sum_k exp(-||z_i - mu_k||^2) ]

Prototypes {mu_k} are running means of labelled examples' latents, one per
known pathology class (Snell, Swersky & Zemel 2017 prototypical-network form).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    """Maintains K class prototypes as an EMA running mean over labelled latents."""

    def __init__(self, num_classes, d_model, momentum=0.9):
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.register_buffer("prototypes", torch.zeros(num_classes, d_model))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, z, labels):
        """z: (B,d), labels: (B,) long in [0, num_classes)."""
        for k in range(self.num_classes):
            mask = labels == k
            if mask.any():
                batch_mean = z[mask].mean(dim=0)
                if not self.initialized[k]:
                    self.prototypes[k] = batch_mean
                    self.initialized[k] = True
                else:
                    self.prototypes[k] = (
                        self.momentum * self.prototypes[k] + (1 - self.momentum) * batch_mean
                    )

    def get(self):
        return self.prototypes  # (K, d)


class GCATT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.prototypes = PrototypeBank(cfg.num_classes, d)

    def cross_attend(self, z_query, mu=None):
        """
        z_query: (B, d) query-crop representation
        mu:      (K, d) source-crop prototype bank (defaults to internal bank)
        returns: (B, d) transferred representation, (B, K) attention weights alpha
        """
        if mu is None:
            mu = self.prototypes.get()

        Wq_z = self.W_q(z_query)               # (B,d)
        Wk_mu = self.W_k(mu)                   # (K,d)
        d = z_query.shape[-1]

        scores = (Wq_z @ Wk_mu.t()) / math.sqrt(d)   # (B,K)
        alpha = scores.softmax(dim=-1)               # (B,K)

        Wv_mu = self.W_v(mu)                    # (K,d)
        transferred = alpha @ Wv_mu              # (B,d)
        return transferred, alpha

    def prototype_loss(self, z, labels):
        """
        L_proto = -log softmax_k(-||z_i - mu_k||^2) at k = p_i
        Implemented as cross-entropy over negative squared distances.
        """
        mu = self.prototypes.get()                       # (K,d)
        dists = torch.cdist(z, mu, p=2) ** 2              # (B,K)
        logits = -dists
        return F.cross_entropy(logits, labels)

    def forward(self, z, labels=None, update_prototypes=True):
        if labels is not None and update_prototypes:
            self.prototypes.update(z.detach(), labels)
        transferred, alpha = self.cross_attend(z)
        loss = self.prototype_loss(z, labels) if labels is not None else None
        return transferred, alpha, loss
