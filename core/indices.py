"""
Per-image descriptive statistics and vegetation indices.

Everything here is computed strictly from the pixels of the image that is
passed in -- nothing is hardcoded or simulated.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage import filters


def _gray(image_array: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 array to grayscale (float)."""
    return np.array(Image.fromarray(image_array).convert("L")).astype(float)


def image_stats(image_array: np.ndarray) -> dict:
    """Real, per-image summary statistics used by several plots."""
    gray = _gray(image_array)
    edges = filters.sobel(gray / 255.0)
    r, g, b = (image_array[:, :, i].astype(float).mean() for i in range(3))
    return {
        "mean_intensity": gray.mean(),
        "std_intensity": gray.std(),
        "edge_density": edges.mean(),
        "edges": edges,
        "r": r, "g": g, "b": b,
    }


def ndvi(image_array: np.ndarray) -> np.ndarray:
    """
    NDVI-style vegetation index.

    NOTE: a standard RGB camera has no true near-infrared (NIR) band, so
    the green channel is used here as a NIR *proxy*. This is a common,
    clearly-labelled approximation -- not a substitute for a real NIR
    sensor.
    """
    red = image_array[:, :, 0].astype(float)
    nir = image_array[:, :, 1].astype(float)  # NIR-proxy (green channel)
    val = (nir - red) / (nir + red + 1e-6)
    return np.clip(val, -1.0, 1.0)


def ndwi(image_array: np.ndarray) -> np.ndarray:
    """NDWI-style water index (blue channel used as NIR-proxy)."""
    green = image_array[:, :, 1].astype(float)
    nir = image_array[:, :, 2].astype(float)  # NIR-proxy (blue channel)
    val = (green - nir) / (green + nir + 1e-6)
    return np.clip(val, -1.0, 1.0)


def feature_vector(image_array: np.ndarray) -> np.ndarray:
    """Compact real descriptor vector per image, used by batch-level charts."""
    stats = image_stats(image_array)
    veg = ndvi(image_array)
    return np.array([
        stats["mean_intensity"], stats["std_intensity"], stats["edge_density"],
        stats["r"], stats["g"], stats["b"],
        float(np.nanmean(veg)), float(np.nanstd(veg)),
    ])


def plot_index_with_colorbar(index_array: np.ndarray, ax, cmap: str, title: str) -> None:
    """Display a vegetation/water index with a properly labeled -1..1 colorbar."""
    import matplotlib.pyplot as plt
    im = ax.imshow(index_array, cmap=cmap, vmin=-1, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{title} (-1 to 1)")
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])


def color_histogram(image_array: np.ndarray, ax) -> None:
    """Real per-image RGB channel histogram, with statistical outlier pixels
    trimmed (values beyond 1.5x IQR per channel) so a few extreme sensor
    artifacts (hot pixels, clipped highlights) don't dominate the scale.
    Also annotates the real measured mean R/G/B values and a chlorophyll
    proxy value directly on the plot."""
    colors = ("red", "green", "blue")
    means = {}
    for i, c in enumerate(colors):
        channel = image_array[:, :, i].ravel().astype(float)
        means[c] = float(channel.mean())
        q1, q3 = np.percentile(channel, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        trimmed = channel[(channel >= lo) & (channel <= hi)]
        ax.hist(trimmed, bins=64, range=(0, 255), color=c, alpha=0.5, label=c.capitalize())

    # Local import to avoid a circular import (vegetation_indices imports from this module).
    from .vegetation_indices import chlorophyll_index
    ci = chlorophyll_index(image_array)
    ci_mean = float(np.mean(ci.array))

    summary = (f"Mean Red   : {means['red']:.1f}\n"
               f"Mean Green : {means['green']:.1f}\n"
               f"Mean Blue  : {means['blue']:.1f}\n"
               f"Chlorophyll Index (proxy, mean): {ci_mean:.2f}")
    ax.text(0.98, 0.97, summary, transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
            family="monospace", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#999999"))

    ax.set_xlabel("Pixel intensity (0-255)")
    ax.set_ylabel("Pixel count")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Colour distribution (outliers beyond 1.5x IQR trimmed)")
