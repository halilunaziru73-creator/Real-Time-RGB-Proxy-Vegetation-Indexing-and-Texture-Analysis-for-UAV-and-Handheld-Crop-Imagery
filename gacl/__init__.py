"""
gacl -- Geometry-Agnostic Contrastive Learning (Section 7 reference implementation)

See README.md in the project root before running anything here. In short:
this code implements the architecture math faithfully. It does NOT by itself
establish that GACL works, since that depends on the label quality and
collection protocol behind whatever data it's run against -- something this
code has no way to verify on its own. evaluate.py computes and prints real
accuracy/F1/AUC/kappa/MCC from GACL's own prototype-distance classification
head (Section 7.5) -- these are genuine metrics, not proxies -- but they are
only as meaningful as the labels they are scored against, and are printed
alongside chance-level context every time so a near-chance result cannot be
mistaken for a validated capability.
"""

from .config import GACLConfig
from .model import GACLModel
from .dataset import GACLTabularDataset, CROP_CLASSES, PATHOLOGY_CLASSES

__all__ = [
    "GACLConfig",
    "GACLModel",
    "GACLTabularDataset",
    "CROP_CLASSES",
    "PATHOLOGY_CLASSES",
]
