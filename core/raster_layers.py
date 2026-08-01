"""
Raster-layer registry for the map canvas.

Each entry provides an array function AND the index's true/declared value
range plus a short label, so the map canvas can render it with a correctly
scaled colormap and a colorbar that actually means something. Each layer
also lists a curated set of scientifically-appropriate colormap
alternatives (symbology) the user can switch between -- diverging palettes
for signed indices, sequential palettes for masks/unbounded surfaces --
rather than an arbitrary single-colour tint that could misrepresent the
measurement.
"""
from __future__ import annotations

import numpy as np
from skimage import filters, morphology

from . import vegetation_indices as vi
from .spectral import band_ratio_map
from .texture import invariant_texture_map, idw_grid

DIVERGING_CMAPS = ["RdYlGn", "RdYlBu_r", "PiYG", "Spectral", "BrBG", "coolwarm"]
SEQUENTIAL_CMAPS = ["Greens", "YlGn", "viridis", "plasma", "Blues", "Reds", "Oranges", "YlOrRd"]
STRETCH_CMAPS = ["viridis", "plasma", "cividis", "magma", "coolwarm"]


def canopy_layer(image_array: np.ndarray) -> np.ndarray:
    r = image_array[:, :, 0].astype(float)
    g = image_array[:, :, 1].astype(float)
    b = image_array[:, :, 2].astype(float)
    total = r + g + b + 1e-6
    exg = 2 * (g / total) - (r / total) - (b / total)
    mask = exg > filters.threshold_otsu(exg)
    mask = morphology.remove_small_objects(mask, min_size=30)
    return mask.astype(float)


def susceptible_layer(image_array: np.ndarray) -> np.ndarray:
    from .indices import ndvi as ndvi_raw
    veg = ndvi_raw(image_array)
    threshold = np.nanpercentile(veg, 20)
    mask = veg <= threshold
    mask = morphology.remove_small_objects(mask, min_size=25)
    return mask.astype(float)


def variable_rate_layer(image_array: np.ndarray) -> np.ndarray:
    from .indices import ndvi as ndvi_raw
    veg = ndvi_raw(image_array)
    zones = np.digitize(veg, bins=[0.0, 0.3, 0.6])
    rate_lookup = np.array([1.5, 1.2, 1.0, 0.7])
    return rate_lookup[zones]


def texture_layer(image_array: np.ndarray) -> np.ndarray:
    return invariant_texture_map(image_array)


def idw_layer(image_array: np.ndarray) -> np.ndarray:
    grid, *_ = idw_grid(image_array)
    return grid  # displayed with its own min-max stretch (not a bounded index)


def _vi_array(func):
    """Wrap a vegetation_indices function to return just the array (or None)."""
    return lambda img: func(img).array


#: (key, display name, cmap_options[list, first=default], array_func, vmin, vmax, unit_label)
RASTER_LAYER_DEFS = [
    ("ndvi", "NDVI (vegetation, RGB-proxy)", DIVERGING_CMAPS, _vi_array(vi.ndvi), -1.0, 1.0, "NDVI (-1 to 1)"),
    ("ndwi", "NDWI (water/moisture, RGB-proxy)", DIVERGING_CMAPS, _vi_array(vi.ndwi), -1.0, 1.0, "NDWI (-1 to 1)"),
    ("gndvi", "GNDVI (RGB-proxy)", DIVERGING_CMAPS, _vi_array(vi.gndvi), -1.0, 1.0, "GNDVI (-1 to 1)"),
    ("evi", "EVI (RGB-proxy)", DIVERGING_CMAPS, _vi_array(vi.evi), -1.0, 1.0, "EVI (-1 to 1, proxy)"),
    ("savi", "SAVI (RGB-proxy)", DIVERGING_CMAPS, _vi_array(vi.savi), -1.0, 1.0, "SAVI (-1 to 1, proxy)"),
    ("msavi", "MSAVI (RGB-proxy)", DIVERGING_CMAPS, _vi_array(vi.msavi), -1.0, 1.0, "MSAVI (-1 to 1, proxy)"),
    ("osavi", "OSAVI (RGB-proxy)", DIVERGING_CMAPS, _vi_array(vi.osavi), -1.0, 1.0, "OSAVI (-1 to 1, proxy)"),
    ("chlorophyll", "Chlorophyll Index (proxy)", SEQUENTIAL_CMAPS, _vi_array(vi.chlorophyll_index), None, None,
     "CI (2nd-98th pct. stretch, proxy)"),
    ("canopy", "Canopy Structure", SEQUENTIAL_CMAPS, canopy_layer, 0.0, 1.0, "Canopy mask (0=bg, 1=canopy)"),
    ("susceptible", "Susceptible Spots (heuristic)", ["Reds"] + SEQUENTIAL_CMAPS, susceptible_layer, 0.0, 1.0,
     "Stress mask (0=normal, 1=flagged)"),
    ("variable_rate", "Variable Rate Zones", ["RdYlBu_r"] + DIVERGING_CMAPS, variable_rate_layer, 0.5, 1.6,
     "Input rate (x baseline)"),
    ("texture", "Invariant Texture", STRETCH_CMAPS, texture_layer, 0.0, 1.0, "Texture magnitude (0-1)"),
    ("idw", "IDW Interpolation", STRETCH_CMAPS, idw_layer, None, None, "Interpolated intensity (min-max stretch)"),
    ("ratio_rg", "Band Ratio R/G", ["coolwarm"] + STRETCH_CMAPS, lambda img: band_ratio_map(img, "R", "G"), None, None,
     "R/G ratio (min-max stretch)"),
    ("variability", "Composite Variability (Std Dev)", ["magma"] + STRETCH_CMAPS, None, None, None,
     "Per-pixel std dev across aligned batch"),
    ("true_ndvi_ms", "True NDVI (UAV multispectral)", DIVERGING_CMAPS, None, -1.0, 1.0,
     "True NDVI (-1 to 1, real NIR band)"),
    ("true_ndre_ms", "True NDRE (UAV multispectral)", DIVERGING_CMAPS, None, -1.0, 1.0,
     "True NDRE (-1 to 1, real red-edge band)"),
]

#: (key, display name, default colour hex) -- simple mock vector overlay.
VECTOR_LAYER_DEFS = [
    ("quadrants", "Quadrant Boundaries", "#d81b60"),
]

#: (key, display name) -- non-raster analyses rendered in the Charts & Reports tab.
REPORT_LAYER_DEFS = [
    ("Cross-Crop Alignment", "Cross-Crop Alignment"),
    ("Geometric Transfer", "Geometric Transfer"),
    ("Loss Functions", "Loss Functions (K-Means)"),
    ("Faster R-CNN", "Faster R-CNN (COCO, generic)"),
    ("ResNet101", "ResNet101 (ImageNet, generic)"),
    ("Pathology Localization", "Pathology Localization"),
    ("Root Architecture", "Root Architecture"),
    ("Colour Histogram", "Colour Histogram (outlier-trimmed)"),
    ("Spectral Reflectance & Band Ratios", "Spectral Reflectance & Band Ratios"),
    ("GLCM Texture Features", "GLCM Texture Features"),
    ("Spectral Index Availability", "Spectral Index Availability (NDRE/PRI/MSI/ARI)"),
    ("Invariance Tests", "Invariance Tests (rotation/scale/flip)"),
    ("InfoNCE (self-supervised)", "Contrastive Loss (InfoNCE, self-supervised)"),
    ("Morphology & Environment Profile", "Morphology & Environment Attribute Profile"),
    ("Time-Series Trend (linear)", "Time-Series Trend (linear)"),
    ("Feature-Space Projection (PCA)", "Feature-Space Projection (PCA)"),
    ("Batch Summary", "Batch Summary"),
    ("Embedding & Similarity Metrics", "Embedding & Similarity Metrics (batch)"),
    ("Cross-Image Pathology Transfer Recommendation", "Cross-Image Pathology Transfer Recommendation"),
    ("Classification / Clustering (automatic)", "Classification / Clustering (automatic)"),
    ("Synthetic Reference Classifier (DEMO)", "Synthetic Crop & Pathology Reference Classifier (SIMULATED DEMO)"),
]
