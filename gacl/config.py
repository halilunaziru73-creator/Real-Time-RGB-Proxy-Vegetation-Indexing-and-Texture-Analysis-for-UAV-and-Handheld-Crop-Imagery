"""
gacl/config.py

Central hyperparameter configuration for the GACL architecture
(Section 7 of the paper: Geometry-Agnostic Contrastive Learning).

NOTE ON STATUS
---------------
GACL has not previously been trained or benchmarked. These defaults are
therefore *architectural* choices only (dimensionality, depth, etc. chosen to
match Section 7's equations), not values tuned against any prior validation
result -- none exists yet to tune against.
"""

from dataclasses import dataclass, field


@dataclass
class HGAViTConfig:
    patch_dim: int = 4          # dim of each pseudo-patch fed to the encoder
    num_patches: int = 8        # embedding_1..32 reshaped into 8x4 pseudo-patches
    d_model: int = 64           # d, latent representation dimension
    num_heads: int = 4
    num_layers: int = 6         # L
    num_scales: int = 3         # S (hierarchical pooling stages)
    geo_dim: int = 5            # camera_height, pitch, roll, yaw, distance
    mlp_ratio: float = 4.0
    dropout: float = 0.1


@dataclass
class GCATTConfig:
    d_model: int = 64
    num_classes: int = 5        # pathology classes: Healthy/Blight/Rust/Mildew/LeafSpot
    temperature: float = 0.1    # used in cross-attention softmax scaling (sqrt(d))


@dataclass
class DHGNNConfig:
    d_model: int = 64
    hidden_dim: int = 64
    num_layers: int = 2
    knn_k: int = 8              # k for k-NN hyperedge construction, per modality


@dataclass
class VLAEConfig:
    z_dim: int = 64             # pathology-relevant latent (shared with HGAViT output)
    eps_dim: int = 16           # environment/nuisance latent (matches latent_z1..16 in data)
    beta: float = 4.0           # beta-VAE disentanglement strength
    decoder_hidden: int = 128
    recon_dim: int = 32         # reconstruct the embedding_1..32 vector as a stand-in for I


@dataclass
class GACLLossWeights:
    lambda_proto: float = 1.0   # lambda_1
    lambda_hyper: float = 0.5   # lambda_2
    lambda_vlae: float = 1.0    # lambda_3
    lambda_ib: float = 0.1      # lambda_4
    lambda_adv: float = 0.1     # lambda_5


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    temperature_geo: float = 0.07   # tau in L_geo
    device: str = "cpu"


@dataclass
class GACLConfig:
    hgavit: HGAViTConfig = field(default_factory=HGAViTConfig)
    gcatt: GCATTConfig = field(default_factory=GCATTConfig)
    dhgnn: DHGNNConfig = field(default_factory=DHGNNConfig)
    vlae: VLAEConfig = field(default_factory=VLAEConfig)
    loss_weights: GACLLossWeights = field(default_factory=GACLLossWeights)
    train: TrainConfig = field(default_factory=TrainConfig)


def make_real_image_config(num_pathology_classes: int = 22, image_size: int = 64,
                            patch_size: int = 8) -> "GACLConfig":
    """
    Config for training GACL on real images via gacl/image_dataset.py, instead
    of the tabular pseudo-patch stand-in (gacl/dataset.py). Dimensions are
    derived directly from the real patchification scheme rather than copied
    from the tabular defaults above.

    NOTE: these are architectural dimension choices only, matching the real
    patch geometry -- not hyperparameters tuned against any prior validated
    result on this dataset (none existed before Section 6.9-6.12 of the paper,
    and that result used classical features, not this image-based model).
    """
    grid = image_size // patch_size
    num_patches = grid * grid
    patch_dim = patch_size * patch_size * 3

    cfg = GACLConfig()
    cfg.hgavit.patch_dim = patch_dim
    cfg.hgavit.num_patches = num_patches
    cfg.hgavit.geo_dim = 5          # kept for architectural compatibility; fed as zeros (see image_dataset.py)
    cfg.gcatt.num_classes = num_pathology_classes
    cfg.vlae.recon_dim = patch_dim  # reconstruction target is the mean patch vector (see image_dataset.py)
    return cfg
