"""
gacl/vlae.py

VLAE -- Variational Latent Agronomic Environment model (Section 7.7).

L_VLAE = E_{q_phi(eps|I)}[log p_psi(I|z,eps)] - beta * D_KL(q_phi(eps|I) || p(eps))

p(eps) = N(0, I); beta follows the beta-VAE formulation (Higgins et al. 2017).
Pathology-relevant information is pushed into z (from HGAViT); illumination /
background nuisance variation is absorbed into eps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VLAE(nn.Module):
    def __init__(self, cfg, input_dim):
        """
        cfg: VLAEConfig
        input_dim: dimensionality of the observation I is reconstructed from.
                    In this repo's tabular stand-in, I is represented by the
                    32-dim embedding vector (see dataset.py / README caveat).
        """
        super().__init__()
        self.cfg = cfg
        self.beta = cfg.beta

        # q_phi(eps | I)
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(128, cfg.eps_dim)
        self.logvar_head = nn.Linear(128, cfg.eps_dim)

        # p_psi(I | z, eps)
        self.dec = nn.Sequential(
            nn.Linear(cfg.z_dim + cfg.eps_dim, cfg.decoder_hidden),
            nn.GELU(),
            nn.Linear(cfg.decoder_hidden, cfg.recon_dim),
        )

    def encode(self, I):
        h = self.enc(I)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps_noise = torch.randn_like(std)
        return mu + eps_noise * std

    def decode(self, z, eps):
        return self.dec(torch.cat([z, eps], dim=-1))

    def forward(self, I, z):
        """
        I: (B, input_dim) observation (or stand-in feature vector)
        z: (B, z_dim) pathology-relevant latent from HGAViT
        returns: I_hat, mu, logvar, loss_vlae (negative ELBO, to be minimized)
        """
        mu, logvar = self.encode(I)
        eps = self.reparameterize(mu, logvar)
        I_hat = self.decode(z, eps)

        recon_loss = F.mse_loss(I_hat, I, reduction="none").sum(dim=-1).mean()
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

        # L_VLAE (paper form is a maximization objective: ELBO - beta*KL);
        # we return its negation as a loss to minimize.
        elbo = -recon_loss  # E_q[log p(I|z,eps)] approximated by -recon MSE
        loss_vlae = -(elbo - self.beta * kl)
        return I_hat, mu, logvar, loss_vlae
