"""Canopy structure, susceptibility screening, variable-rate, XAI panels."""
from __future__ import annotations

import numpy as np
from skimage import filters, measure, morphology

from .indices import image_stats, ndvi


def vis_nir_graph(image_array: np.ndarray, ax) -> None:
    """
    Real per-image VIS vs NIR-proxy comparison. Standard RGB cameras have
    no NIR band, so (consistent with the NDVI/NDWI proxies elsewhere) the
    green and blue channels stand in for NIR here.
    """
    r = image_array[:, :, 0].astype(float).mean()
    g = image_array[:, :, 1].astype(float).mean()
    b = image_array[:, :, 2].astype(float).mean()
    vis_mean = (r + g + b) / 3.0
    labels = ["VIS mean\n(R+G+B)/3", "NIR-proxy\n(NDVI channel)", "NIR-proxy\n(NDWI channel)"]
    values = [vis_mean, g, b]
    ax.bar(labels, values, color=["gray", "darkgreen", "navy"])
    ax.set_ylabel("Mean intensity (0-255)")
    ax.set_title("VIS vs NIR-proxy (RGB camera — no true NIR band)")


def xai_recommendation(image_array: np.ndarray, ax) -> None:
    """Rule-based interpretability summary derived from this image's own statistics."""
    stats = image_stats(image_array)
    veg = ndvi(image_array)
    mean_ndvi = float(np.nanmean(veg))
    susceptible_frac = float(np.mean(veg <= np.nanpercentile(veg, 20)))
    edge_density = stats["edge_density"]

    lines = [
        f"Mean intensity       : {stats['mean_intensity']:.2f}",
        f"Edge density         : {edge_density:.4f}  {'(elevated)' if edge_density > 0.15 else '(normal)'}",
        f"Mean NDVI            : {mean_ndvi:.3f}  {'(low / possible stress)' if mean_ndvi < 0.2 else '(normal range)'}",
        f"Low-vigor pixel share: {susceptible_frac * 100:.1f}%",
    ]
    flagged = susceptible_frac > 0.15 or mean_ndvi < 0.2 or edge_density > 0.15
    recommendation = ("prioritise field inspection of flagged zones."
                       if flagged else "routine monitoring sufficient at this time.")

    text = (
        "XAI-style recommendation (rule-based, computed fresh from\n"
        "this image's own measured statistics):\n\n" + "\n".join(lines) +
        f"\n\nRecommendation: {recommendation}"
    )
    ax.axis("off")
    ax.text(0.02, 0.5, text, fontsize=8, va="center", family="monospace", wrap=True)


def variable_rate_map(image_array: np.ndarray, ax) -> None:
    """
    Standard NDVI-based variable-rate zone map (a common precision-
    agriculture heuristic), computed strictly from this image's own NDVI.
    """
    import matplotlib.pyplot as plt
    veg = ndvi(image_array)
    zones = np.digitize(veg, bins=[0.0, 0.3, 0.6])  # 0,1,2,3
    rate_lookup = np.array([1.5, 1.2, 1.0, 0.7])
    rate_map = rate_lookup[zones]
    im = ax.imshow(rate_map, cmap="RdYlBu_r", vmin=0.5, vmax=1.6)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Illustrative input rate (x baseline)")
    ax.set_title("NDVI-based Variable Rate Zone Map\n(standard precision-ag heuristic)")
    ax.axis("off")


def pathology_localization_demo(image_array: np.ndarray, ax) -> None:
    """Row-wise mean edge intensity, derived from this image's real edge map."""
    stats = image_stats(image_array)
    edges = stats["edges"]
    n_samples = 6
    rows = np.linspace(0, edges.shape[0] - 1, n_samples).astype(int)
    scores = [edges[r, :].mean() for r in rows]
    ax.plot(range(1, n_samples + 1), scores, marker="o")
    ax.set_xlabel("Sample (image row band)")
    ax.set_ylabel("Localization Score (mean edge response)")


def susceptible_spot_detection(image_array: np.ndarray, ax) -> None:
    """
    Heuristic susceptibility screen: flags regions with low vegetation
    index as potentially stressed/diseased canopy. A simple threshold +
    connected-component heuristic, not a published pathology model.
    """
    import matplotlib.pyplot as plt
    veg_index = ndvi(image_array)
    threshold = np.nanpercentile(veg_index, 20)  # bottom 20% of vegetation signal
    stressed_mask = veg_index <= threshold
    stressed_mask = morphology.remove_small_objects(stressed_mask, min_size=25)

    ax.imshow(image_array)
    labeled = measure.label(stressed_mask)
    props = measure.regionprops(labeled)
    for p in props:
        minr, minc, maxr, maxc = p.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor="red", linewidth=1.2)
        ax.add_patch(rect)
    ax.set_title(f"Susceptible spots (heuristic): {len(props)} region(s) flagged")
    ax.axis("off")


def canopy_structure(image_array: np.ndarray, ax) -> None:
    """Real canopy segmentation using an Excess Green Index (ExG)."""
    r = image_array[:, :, 0].astype(float)
    g = image_array[:, :, 1].astype(float)
    b = image_array[:, :, 2].astype(float)
    total = r + g + b + 1e-6
    exg = 2 * (g / total) - (r / total) - (b / total)
    canopy_mask = exg > filters.threshold_otsu(exg)
    canopy_mask = morphology.remove_small_objects(canopy_mask, min_size=30)
    cover_pct = 100.0 * canopy_mask.sum() / canopy_mask.size

    ax.imshow(canopy_mask, cmap="Greens")
    ax.set_title(f"Canopy structure: {cover_pct:.1f}% cover")
    ax.axis("off")
