"""Batch-level (cross-image) panels: PCA feature space, linear trend, summary."""
from __future__ import annotations

import numpy as np
from skimage import filters

from .indices import feature_vector, ndvi

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def plot_feature_space(loaded_images: list[np.ndarray], ax) -> None:
    """
    Real 2D PCA projection of each image's own colour/texture/NDVI
    descriptors. This is a genuine, descriptive feature-space
    visualisation -- no contrastive-learning model is trained anywhere
    in this pipeline.
    """
    if not SKLEARN_AVAILABLE:
        ax.axis("off")
        ax.text(0.3, 0.5, "PCA unavailable: scikit-learn not installed.", fontsize=9)
        return
    if len(loaded_images) < 2:
        ax.axis("off")
        ax.text(0.2, 0.5, "Feature-space projection needs at least 2 images.", fontsize=10)
        return
    feats = np.array([feature_vector(im) for im in loaded_images])
    feats = feats - feats.mean(axis=0)
    n_components = min(2, feats.shape[0] - 1, feats.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(feats)
    if n_components == 1:
        coords = np.hstack([coords, np.zeros_like(coords)])
    var = list(pca.explained_variance_ratio_) + [0, 0]

    ax.scatter(coords[:, 0], coords[:, 1], c=range(len(loaded_images)), cmap="viridis", s=90, edgecolor="k")
    for i, (x, y) in enumerate(coords[:, :2]):
        ax.annotate(f"Img {i + 1}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}% variance)")
    ax.set_title("Feature-space projection — real PCA on colour/texture/NDVI descriptors")


def plot_batch_summary(loaded_images: list[np.ndarray], ax) -> None:
    """Real aggregate statistics and a brief data-driven conclusion across the batch."""
    ndvis = [float(np.nanmean(ndvi(im))) for im in loaded_images]
    covers = []
    for im in loaded_images:
        r, g, b = im[:, :, 0].astype(float), im[:, :, 1].astype(float), im[:, :, 2].astype(float)
        total = r + g + b + 1e-6
        exg = 2 * (g / total) - (r / total) - (b / total)
        mask = exg > filters.threshold_otsu(exg)
        covers.append(100.0 * mask.sum() / mask.size)

    lines = [
        f"Images analysed    : {len(loaded_images)}",
        f"Mean NDVI (batch)  : {np.mean(ndvis):.3f}  (range {min(ndvis):.3f} to {max(ndvis):.3f})",
        f"Mean canopy cover  : {np.mean(covers):.1f}%  (range {min(covers):.1f}% to {max(covers):.1f}%)",
    ]
    lowest_i, highest_i = int(np.argmin(ndvis)) + 1, int(np.argmax(ndvis)) + 1
    conclusion = (f"Image {lowest_i} shows the lowest vegetation signal in this batch and "
                  f"Image {highest_i} the highest; canopy cover spans "
                  f"{max(covers) - min(covers):.1f} percentage points across the set.")
    text = "Batch Summary (computed from all processed images):\n\n" + "\n".join(lines) + "\n\nBrief conclusion: " + conclusion
    ax.axis("off")
    ax.text(0.02, 0.5, text, fontsize=9, va="center", family="monospace", wrap=True)


def plot_time_series_trend(loaded_images: list[np.ndarray], ax) -> None:
    """
    Honest statistical trend across the selected image batch: mean NDVI
    per image with a fitted linear regression and a short extrapolation.
    This is NOT a trained deep-learning forecaster -- that would need a
    real historical dataset and a trained network, which isn't available
    here.
    """
    x = np.arange(1, len(loaded_images) + 1)
    y = np.array([np.nanmean(ndvi(img)) for img in loaded_images])

    coeffs = np.polyfit(x, y, deg=1)
    trend_line = np.poly1d(coeffs)

    x_future = np.arange(1, len(loaded_images) + 3)  # extrapolate 2 steps
    ax.plot(x, y, "o-", color="green", label="Measured mean NDVI")
    ax.plot(x_future, trend_line(x_future), "--", color="gray",
            label=f"Linear trend (slope={coeffs[0]:.4f}/step)")
    ax.axvspan(len(loaded_images) + 0.5, x_future[-1] + 0.5, color="gray", alpha=0.08)
    ax.set_xlabel("Image sequence (time step)")
    ax.set_ylabel("Mean NDVI")
    ax.set_ylim(-1, 1)
    ax.set_title("Time-Series Trend — linear extrapolation")
    ax.legend(fontsize=8)


# ------------------------------------------------------------------ #
# Lightweight, descriptor-only variants.
#
# For batches of up to ~2000 images, keeping every full-resolution image
# in memory simultaneously isn't feasible. These variants work from the
# small (8-value) descriptor already computed for each image during
# alignment (core.alignment.ImageRecord.descriptor), so cross-image
# comparisons stay available without holding thousands of full images.
# Descriptor layout (see core.indices.feature_vector):
#   [mean_intensity, std_intensity, edge_density, r, g, b, mean_ndvi, std_ndvi]
# ------------------------------------------------------------------ #
NDVI_DESCRIPTOR_INDEX = 6


def plot_feature_space_from_records(names: list[str], descriptors: np.ndarray, ax) -> None:
    """PCA feature-space projection from pre-computed per-image descriptors."""
    if not SKLEARN_AVAILABLE:
        ax.axis("off")
        ax.text(0.3, 0.5, "PCA unavailable: scikit-learn not installed.", fontsize=9)
        return
    if len(names) < 2:
        ax.axis("off")
        ax.text(0.2, 0.5, "Feature-space projection needs at least 2 images.", fontsize=10)
        return
    feats = descriptors - descriptors.mean(axis=0)
    n_components = min(2, feats.shape[0] - 1, feats.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(feats)
    if n_components == 1:
        coords = np.hstack([coords, np.zeros_like(coords)])
    var = list(pca.explained_variance_ratio_) + [0, 0]

    ax.scatter(coords[:, 0], coords[:, 1], c=range(len(names)), cmap="viridis",
               s=max(10, 200 / max(1, len(names) ** 0.5)), edgecolor="k", linewidths=0.3)
    if len(names) <= 40:  # avoid unreadable label clutter for large batches
        for i, (x, y) in enumerate(coords[:, :2]):
            ax.annotate(f"{i + 1}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=6)
    ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}% variance)")
    ax.set_title(f"Feature-space projection — real PCA on {len(names)} images' descriptors")


def plot_batch_summary_from_records(names: list[str], descriptors: np.ndarray, ax) -> None:
    """Real aggregate statistics across the batch, from pre-computed descriptors."""
    ndvis = descriptors[:, NDVI_DESCRIPTOR_INDEX]
    lines = [
        f"Images analysed    : {len(names)}",
        f"Mean NDVI (batch)  : {np.mean(ndvis):.3f}  (range {ndvis.min():.3f} to {ndvis.max():.3f})",
        f"Mean intensity     : {descriptors[:, 0].mean():.2f}",
        f"Mean edge density  : {descriptors[:, 2].mean():.4f}",
    ]
    lowest_i, highest_i = int(np.argmin(ndvis)), int(np.argmax(ndvis))
    conclusion = (f"'{names[lowest_i]}' shows the lowest vegetation signal in this batch and "
                  f"'{names[highest_i]}' the highest.")
    text = "Batch Summary (computed from all processed images):\n\n" + "\n".join(lines) + "\n\nBrief conclusion: " + conclusion
    ax.axis("off")
    ax.text(0.02, 0.5, text, fontsize=9, va="center", family="monospace", wrap=True)


def plot_time_series_trend_from_records(names: list[str], descriptors: np.ndarray, ax) -> None:
    """Linear trend across the batch sequence, from pre-computed descriptors."""
    x = np.arange(1, len(names) + 1)
    y = descriptors[:, NDVI_DESCRIPTOR_INDEX]

    coeffs = np.polyfit(x, y, deg=1)
    trend_line = np.poly1d(coeffs)

    x_future = np.arange(1, len(names) + 3)
    ax.plot(x, y, "o-", color="green", markersize=3, label="Measured mean NDVI")
    ax.plot(x_future, trend_line(x_future), "--", color="gray",
            label=f"Linear trend (slope={coeffs[0]:.5f}/step)")
    ax.axvspan(len(names) + 0.5, x_future[-1] + 0.5, color="gray", alpha=0.08)
    ax.set_xlabel("Image sequence (load order)")
    ax.set_ylabel("Mean NDVI")
    ax.set_ylim(-1, 1)
    ax.set_title("Time-Series Trend — linear extrapolation")
    ax.legend(fontsize=8)
