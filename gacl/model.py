"""
gacl/model.py

GACLModel -- wires HGAViT, GCATT, DHGNN and VLAE together per the composite
objective of Section 7.8:

    L_GACL = L_geo + lambda_1 L_proto + lambda_2 L_hypergraph
             + lambda_3 L_VLAE + lambda_4 L_IB - lambda_5 L_adv
"""

import torch
import torch.nn as nn

from .hgavit import HGAViT
from .gcatt import GCATT
from .dhgnn import DHGNN
from .vlae import VLAE
from .losses import (
    info_nce_geo,
    hypergraph_node_ce,
    information_bottleneck_bound,
    adversarial_domain_loss,
    composite_gacl_loss,
    DomainClassifier,
)


class GACLModel(nn.Module):
    def __init__(self, cfg, num_crops, recon_input_dim):
        super().__init__()
        self.cfg = cfg
        self.encoder = HGAViT(cfg.hgavit)
        self.gcatt = GCATT(cfg.gcatt)
        self.dhgnn = DHGNN(cfg.dhgnn)
        self.vlae = VLAE(cfg.vlae, input_dim=recon_input_dim)
        self.domain_clf = DomainClassifier(cfg.hgavit.d_model, num_crops)

        # simple geometry perturbation used to build the InfoNCE positive pair;
        # additive noise on the geometry descriptor, standing in for the
        # rotation/tilt/scale augmentation applied at training time (Sec 7.4)
        self.geo_perturb_std = 0.1

    def encode(self, x, g):
        return self.encoder(x, g)

    def forward(self, batch, weights, tau_geo=0.07):
        """
        batch: dict with keys
            'patches' (B,N,patch_dim), 'geo' (B,geo_dim), 'recon_target' (B,recon_dim),
            'pathology' (B,) long, 'crop' (B,) long
        """
        x = batch["patches"]
        g = batch["geo"]
        pathology = batch["pathology"]
        crop = batch["crop"]
        recon_target = batch["recon_target"]

        # anchor pass
        z, _ = self.encode(x, g)

        # geometry-perturbed positive pass (Section 7.4)
        g_pos = g + self.geo_perturb_std * torch.randn_like(g)
        z_pos, _ = self.encode(x, g_pos)

        l_geo = info_nce_geo(z, z_pos, temperature=tau_geo)

        # GCATT: cross-attention transfer + prototype-alignment loss (7.5)
        transferred, alpha, l_proto = self.gcatt(z, labels=pathology)

        # DHGNN: dynamic hypergraph conv + node-classification CE (7.6, 7.8)
        hyper_out, hyper_logits = self.dhgnn(z, crop, pathology, g)
        l_hyper = hypergraph_node_ce(hyper_logits, torch.zeros_like(pathology))
        # NOTE: DHGNN's classifier head is unsupervised-by-default in this
        # reference implementation (no ground-truth hyperedge label exists);
        # see README for what real supervision would require.

        # VLAE (7.7)
        I_hat, mu_eps, logvar_eps, l_vlae = self.vlae(recon_target, z)

        # Information bottleneck bound (7.8)
        l_ib = information_bottleneck_bound(mu_eps, logvar_eps, hyper_logits,
                                             torch.zeros_like(pathology))

        # Adversarial domain-adaptation term (7.8)
        domain_logits = self.domain_clf(z)
        l_adv = adversarial_domain_loss(domain_logits, crop)

        l_total = composite_gacl_loss(l_geo, l_proto, l_hyper, l_vlae, l_ib, l_adv, weights)

        return {
            "loss_total": l_total,
            "loss_geo": l_geo,
            "loss_proto": l_proto,
            "loss_hyper": l_hyper,
            "loss_vlae": l_vlae,
            "loss_ib": l_ib,
            "loss_adv": l_adv,
            "z": z,
            "transferred": transferred,
            "alpha": alpha,
        }
