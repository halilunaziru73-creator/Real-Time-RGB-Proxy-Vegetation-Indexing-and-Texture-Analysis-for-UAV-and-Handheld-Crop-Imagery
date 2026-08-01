"""
Vegetation indices.

IMPORTANT HONESTY NOTE
-----------------------
A standard RGB photo has three bands: Red, Green, Blue. It has NO
near-infrared (NIR), red-edge, or shortwave-infrared (SWIR) band. Every
index below that classically requires NIR is computed here using the
green channel as a documented *stand-in* for NIR (the same convention
already used elsewhere in this pipeline for NDVI/NDWI) -- this is a
common, clearly-labelled RGB approximation used in citizen-science and
low-cost sensing tools, NOT a substitute for a calibrated multispectral
sensor. Indices that fundamentally require a band RGB does not have at
all (red-edge for true NDRE/ARI, narrow 531/570nm bands for PRI, SWIR for
MSI) are marked UNAVAILABLE rather than faked with a look-alike formula.

Every function returns a value/array plus (where relevant) the true
theoretical range, so callers can label colourbars and text panels
correctly instead of showing an unscaled, meaningless picture.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .indices import ndvi as _ndvi_core
from .indices import ndwi as _ndwi_core


@dataclass
class IndexResult:
    name: str
    array: np.ndarray | None      # 2D map, or None if unavailable
    value_range: tuple[float, float]
    is_proxy: bool
    note: str


def _channels(image_array: np.ndarray):
    r = image_array[:, :, 0].astype(float)
    g = image_array[:, :, 1].astype(float)      # used as NIR-proxy where required
    b = image_array[:, :, 2].astype(float)
    return r, g, b


def ndvi(image_array: np.ndarray) -> IndexResult:
    """Normalized Difference Vegetation Index (NIR-proxy = green channel)."""
    return IndexResult("NDVI", _ndvi_core(image_array), (-1.0, 1.0), True,
                        "RGB proxy: green channel stands in for NIR. Not a calibrated spectral measurement.")


def ndwi(image_array: np.ndarray) -> IndexResult:
    """Normalized Difference Water Index (green/blue channels used as NIR-proxy)."""
    return IndexResult("NDWI", _ndwi_core(image_array), (-1.0, 1.0), True,
                        "RGB proxy: blue channel stands in for NIR. Not a calibrated spectral measurement.")


def gndvi(image_array: np.ndarray) -> IndexResult:
    """Green NDVI: (NIR-proxy - Green) / (NIR-proxy + Green). Degenerates toward 0
    under this pipeline's single green-as-NIR convention, so blue is used as the
    NIR-proxy here instead to keep the two indices distinguishable -- still an
    approximation, not a genuine NIR measurement."""
    r, g, b = _channels(image_array)
    nir_proxy = b
    val = (nir_proxy - g) / (nir_proxy + g + 1e-6)
    return IndexResult("GNDVI", np.clip(val, -1, 1), (-1.0, 1.0), True,
                        "RGB proxy: blue channel stands in for NIR. Not a calibrated spectral measurement.")


def evi(image_array: np.ndarray) -> IndexResult:
    """Enhanced Vegetation Index (RGB-proxy form; NIR-proxy = green channel)."""
    r, g, b = _channels(image_array)
    nir_proxy = g
    r_n, g_n, b_n = r / 255.0, nir_proxy / 255.0, b / 255.0
    val = 2.5 * (g_n - r_n) / (g_n + 6 * r_n - 7.5 * b_n + 1 + 1e-6)
    return IndexResult("EVI", np.clip(val, -1, 1), (-1.0, 1.0), True,
                        "RGB proxy formula (NIR-proxy = green). Not a calibrated spectral measurement.")


def savi(image_array: np.ndarray, L: float = 0.5) -> IndexResult:
    """Soil-Adjusted Vegetation Index (RGB-proxy form)."""
    r, g, b = _channels(image_array)
    nir_proxy = g / 255.0
    r_n = r / 255.0
    val = ((nir_proxy - r_n) / (nir_proxy + r_n + L + 1e-6)) * (1 + L)
    return IndexResult("SAVI", np.clip(val, -1, 1), (-1.0, 1.0), True,
                        "RGB proxy formula (NIR-proxy = green, L=0.5). Not a calibrated spectral measurement.")


def msavi(image_array: np.ndarray) -> IndexResult:
    """Modified Soil-Adjusted Vegetation Index (RGB-proxy form)."""
    r, g, b = _channels(image_array)
    nir_proxy = g / 255.0
    r_n = r / 255.0
    val = (2 * nir_proxy + 1 - np.sqrt((2 * nir_proxy + 1) ** 2 - 8 * (nir_proxy - r_n))) / 2
    return IndexResult("MSAVI", np.clip(val, -1, 1), (-1.0, 1.0), True,
                        "RGB proxy formula (NIR-proxy = green). Not a calibrated spectral measurement.")


def osavi(image_array: np.ndarray) -> IndexResult:
    """Optimized Soil-Adjusted Vegetation Index (RGB-proxy form)."""
    r, g, b = _channels(image_array)
    nir_proxy = g / 255.0
    r_n = r / 255.0
    val = (nir_proxy - r_n) / (nir_proxy + r_n + 0.16 + 1e-6)
    return IndexResult("OSAVI", np.clip(val, -1, 1), (-1.0, 1.0), True,
                        "RGB proxy formula (NIR-proxy = green). Not a calibrated spectral measurement.")


def chlorophyll_index(image_array: np.ndarray) -> IndexResult:
    """Chlorophyll Index, CI = (NIR-proxy / Green) - 1 -- here NIR-proxy = blue
    to avoid the green/green degeneracy, so treat as a rough approximation only.
    Green is floored to avoid a division blow-up on near-black pixels."""
    r, g, b = _channels(image_array)
    g_safe = np.maximum(g, 8.0)  # avoid extreme ratios on near-zero green pixels
    val = (b / g_safe) - 1
    val = np.clip(val, -5.0, 20.0)
    lo, hi = float(np.percentile(val, 2)), float(np.percentile(val, 98))
    if lo == hi:
        lo, hi = float(val.min()), float(val.max() + 1e-6)
    return IndexResult("Chlorophyll Index (CI)", val, (lo, hi), True,
                        "RGB proxy: no real chlorophyll-sensitive band present. Indicative only; displayed with a "
                        "2nd-98th percentile stretch, not a fixed physical range.")


def ndre_unavailable() -> IndexResult:
    return IndexResult("NDRE", None, (-1.0, 1.0), False,
                        "UNAVAILABLE: true NDRE requires a red-edge band (~705-750nm) that no RGB camera captures. "
                        "No approximation is shown to avoid implying a real red-edge measurement.")


def pri_unavailable() -> IndexResult:
    return IndexResult("PRI", None, (-1.0, 1.0), False,
                        "UNAVAILABLE: PRI requires narrowband reflectance at 531nm and 570nm from a hyperspectral "
                        "sensor. Not derivable from RGB photography.")


def msi_unavailable() -> IndexResult:
    return IndexResult("MSI", None, (0.0, 3.0), False,
                        "UNAVAILABLE: Moisture Stress Index requires a shortwave-infrared (SWIR ~1600nm) band. "
                        "NDWI (RGB-proxy) is the closest available moisture-related index in this pipeline.")


def ari_unavailable() -> IndexResult:
    return IndexResult("ARI", None, (0.0, 10.0), False,
                        "UNAVAILABLE: true Anthocyanin Reflectance Index requires a genuine red-edge band. "
                        "Not derivable from RGB photography.")


#: All spatial (map-able) indices, used to build the map-canvas raster layer registry.
SPATIAL_INDEX_FUNCS = {
    "NDVI": ndvi,
    "NDWI": ndwi,
    "GNDVI": gndvi,
    "EVI": evi,
    "SAVI": savi,
    "MSAVI": msavi,
    "OSAVI": osavi,
    "Chlorophyll Index (CI)": chlorophyll_index,
}

#: Indices that cannot be honestly computed from RGB input at all.
UNAVAILABLE_INDEX_FUNCS = {
    "NDRE": ndre_unavailable,
    "PRI": pri_unavailable,
    "MSI": msi_unavailable,
    "ARI": ari_unavailable,
}
