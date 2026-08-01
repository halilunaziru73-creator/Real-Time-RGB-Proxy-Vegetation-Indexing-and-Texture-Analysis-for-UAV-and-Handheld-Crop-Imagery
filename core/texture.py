"""Texture, cross-crop alignment, quantization-loss, IDW, root-architecture panels."""
from __future__ import annotations

import numpy as np
from skimage import color, filters, morphology

from .indices import _gray, image_stats

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def invariant_texture_map(image_array: np.ndarray) -> np.ndarray:
    """Gradient-magnitude texture map, normalized to [0, 1]."""
    gray = _gray(image_array)
    texture = np.abs(np.gradient(gray))
    texture_map = np.sqrt(texture[0] ** 2 + texture[1] ** 2)
    return texture_map / (np.max(texture_map) + 1e-9)


def cross_crop_alignment_matrix(image_array: np.ndarray):
    """
    Real quadrant-similarity matrix: splits the image into four regions and
    compares their grayscale histograms (correlation coefficient).
    """
    gray = _gray(image_array)
    h, w = gray.shape
    quads = {
        "Chestnut grove": gray[:h // 2, :w // 2],
        "Maize": gray[:h // 2, w // 2:],
        "Cotton": gray[h // 2:, :w // 2],
        "Rice": gray[h // 2:, w // 2:],
    }
    labels = list(quads.keys())
    hists = [np.histogram(q, bins=32, range=(0, 255))[0].astype(float) for q in quads.values()]
    hists = [h / (h.sum() + 1e-9) for h in hists]

    n = len(labels)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = np.corrcoef(hists[i], hists[j])[0, 1]
    return labels, matrix


def geometric_transfer_metrics(image_array: np.ndarray, ax) -> None:
    """Genuine per-image descriptive statistics panel (not a benchmark claim)."""
    stats = image_stats(image_array)
    ax.axis("off")
    text = (
        "Per-Image Metrics (computed):\n"
        f"  Mean intensity      : {stats['mean_intensity']:.2f}\n"
        f"  Intensity std dev   : {stats['std_intensity']:.2f}\n"
        f"  Edge density        : {stats['edge_density']:.4f}\n"
        f"  Channel means (R,G,B): {stats['r']:.1f}, {stats['g']:.1f}, {stats['b']:.1f}\n\n"
        "Note: these are descriptive statistics of this specific image,\n"
        "not validated cross-dataset benchmark results."
    )
    ax.text(0.05, 0.5, text, fontsize=10, va="center", family="monospace")


def image_quantization_loss(image_array: np.ndarray, ax) -> None:
    """Real K-Means colour-quantization reconstruction-loss curve for this image."""
    if not SKLEARN_AVAILABLE:
        ax.axis("off")
        ax.text(0.05, 0.5, "Loss curve unavailable: scikit-learn not installed.\npip install scikit-learn",
                 fontsize=9, va="center", family="monospace")
        return
    pixels = image_array.reshape(-1, 3).astype(float)
    rng = np.random.default_rng(0)
    if pixels.shape[0] > 4000:
        idx = rng.choice(pixels.shape[0], 4000, replace=False)
        pixels = pixels[idx]
    ks = list(range(1, 9))
    losses = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(pixels)
        losses.append(km.inertia_)
    ax.plot(ks, losses, marker="o", color="crimson")
    ax.set_xlabel("k (colour clusters)")
    ax.set_ylabel("K-Means inertia (reconstruction loss)")
    ax.set_title("Per-image colour-quantization loss curve")


def idw_grid(image_array: np.ndarray, n_points: int = 15, grid_size: int = 60):
    """
    Compute the raw inverse-distance-weighted interpolation grid for an
    image, seeded deterministically by the image's own pixel content.
    Returns (grid, xs, ys, vals, w, h) so callers can both plot it (with
    sample points) and reuse the grid alone as a raster overlay.
    """
    gray = _gray(image_array)
    h, w = gray.shape
    rng = np.random.default_rng(seed=int(gray.sum()) % (2 ** 32))
    ys = rng.integers(0, h, n_points)
    xs = rng.integers(0, w, n_points)
    vals = gray[ys, xs]

    grid_x, grid_y = np.meshgrid(np.linspace(0, w - 1, grid_size), np.linspace(0, h - 1, grid_size))
    # Vectorized IDW: broadcast (grid_size, grid_size, n_points) distances at once
    # instead of a Python-level double loop over every grid cell.
    dx = grid_x[:, :, None] - xs[None, None, :]
    dy = grid_y[:, :, None] - ys[None, None, :]
    dists = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    weights = 1.0 / dists ** 2
    grid = (weights * vals[None, None, :]).sum(axis=2) / weights.sum(axis=2)
    return grid, xs, ys, vals, w, h


def idw_demo(image_array: np.ndarray, ax) -> None:
    """Real inverse-distance-weighted interpolation over sampled pixel intensities."""
    import matplotlib.pyplot as plt
    grid, xs, ys, vals, w, h = idw_grid(image_array)
    im = ax.imshow(grid, cmap="plasma", origin="upper")
    ax.scatter((xs / w) * grid.shape[1], (ys / h) * grid.shape[0], c="cyan", s=15, label="sample pts")
    ax.legend(loc="upper right", fontsize=7)
    plt.colorbar(im, ax=ax)


def root_architecture(image_array: np.ndarray, ax) -> None:
    """Sobel-edge + skeletonization view (root/vein-like structure)."""
    gray = color.rgb2gray(image_array)
    edges = filters.sobel(gray)
    binary = edges > np.mean(edges)
    skeleton = morphology.skeletonize(binary)
    ax.imshow(skeleton, cmap="gray")
