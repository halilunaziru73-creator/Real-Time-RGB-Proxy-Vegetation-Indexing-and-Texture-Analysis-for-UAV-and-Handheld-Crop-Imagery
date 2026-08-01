"""
gacl/losses.py

Section 7.4: geometry-invariance InfoNCE loss L_geo
Section 7.8: composite objective L_GACL and its remaining terms
             (L_hypergraph, L_IB, L_adv), plus the GradReverse layer used by
             the adversarial domain-adaptation term (Ganin & Lempitsky 2015).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_geo(z, z_pos, temperature=0.07):
    """
    L_geo = -(1/B) sum_i log [ exp(cos(z_i,z_i^+)/tau) /
                                sum_j exp(cos(z_i,z_j)/tau) ]

    z, z_pos: (B, d) anchor and geometry-perturbed positive representations.
    In-batch negatives, SimCLR-style (Chen et al. 2020).
    """
    z = F.normalize(z, dim=-1)
    z_pos = F.normalize(z_pos, dim=-1)

    sim = z @ z_pos.t() / temperature       # (B,B); row i vs all positives j
    labels = torch.arange(z.shape[0], device=z.device)
    loss = F.cross_entropy(sim, labels)
    return loss


class GradReverse(torch.autograd.Function):
    """Gradient reversal layer for the adversarial domain-adaptation term."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class DomainClassifier(nn.Module):
    """Predicts crop identity c_i from z; trained adversarially via GradReverse
    so that z is discouraged from encoding crop identity (only pathology
    should transfer, Section 7.8)."""

    def __init__(self, d_model, num_crops):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, num_crops),
        )

    def forward(self, z, lambd=1.0):
        z_rev = grad_reverse(z, lambd)
        return self.net(z_rev)


def adversarial_domain_loss(domain_logits, crop_labels):
    """L_adv: standard cross-entropy on crop identity, combined with grad-reversal
    upstream so minimizing this term still (via reversal) discourages crop
    leakage into z during the backward pass."""
    return F.cross_entropy(domain_logits, crop_labels)


def hypergraph_node_ce(logits, labels):
    """L_hypergraph: node-classification cross-entropy over DHGNN output Z^(L)."""
    return F.cross_entropy(logits, labels)


def information_bottleneck_bound(mu, logvar, logits, labels, gamma=1.0):
    """
    L_IB = I(z;I) - gamma * I(z;p), approximated via a variational
    upper/lower bound (Alemi et al. 2017 deep-VIB form):

      - I(z;I) upper-bounded by KL(q(z|I) || r(z)) with r(z) = N(0,I)
      - I(z;p) lower-bounded by the (negative) classification cross-entropy
    """
    kl_upper = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    ce_lower = F.cross_entropy(logits, labels)  # proxy lower bound on I(z;p)
    return kl_upper - gamma * (-ce_lower)


def composite_gacl_loss(l_geo, l_proto, l_hyper, l_vlae, l_ib, l_adv, weights):
    """
    L_GACL = L_geo + lambda_1 L_proto + lambda_2 L_hypergraph + lambda_3 L_VLAE
             + lambda_4 L_IB - lambda_5 L_adv
    """
    total = (
        l_geo
        + weights.lambda_proto * l_proto
        + weights.lambda_hyper * l_hyper
        + weights.lambda_vlae * l_vlae
        + weights.lambda_ib * l_ib
        - weights.lambda_adv * l_adv
    )
    return total
