"""
Real texture features via the Grey-Level Co-occurrence Matrix (GLCM).

These are genuine, well-established texture descriptors (Haralick
features) computed directly from each image's own pixels -- not proxies,
not simulated.
"""
from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops

from .indices import _gray


def glcm_features(image_array: np.ndarray, distances=(1,), angles=(0,)) -> dict:
    """
    Compute standard Haralick/GLCM texture features: contrast, dissimilarity,
    homogeneity, energy, correlation, and angular second moment (ASM).
    """
    gray = _gray(image_array).astype(np.uint8)
    # Reduce grey levels for a tractable, fast co-occurrence matrix.
    levels = 32
    quantized = (gray.astype(float) / 256.0 * levels).astype(np.uint8)
    quantized = np.clip(quantized, 0, levels - 1)

    glcm = graycomatrix(quantized, distances=list(distances), angles=list(angles),
                         levels=levels, symmetric=True, normed=True)

    return {
        "contrast": float(graycoprops(glcm, "contrast").mean()),
        "dissimilarity": float(graycoprops(glcm, "dissimilarity").mean()),
        "homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "energy": float(graycoprops(glcm, "energy").mean()),
        "correlation": float(graycoprops(glcm, "correlation").mean()),
        "ASM": float(graycoprops(glcm, "ASM").mean()),
    }


def glcm_report(image_array: np.ndarray, ax) -> None:
    """Render GLCM texture features as a labelled text panel."""
    feats = glcm_features(image_array)
    lines = ["GLCM Texture Features (real, computed from this image):", ""]
    for name, value in feats.items():
        lines.append(f"  {name:14s}: {value:.4f}")
    ax.axis("off")
    ax.text(0.02, 0.5, "\n".join(lines), fontsize=9, va="center", family="monospace")
