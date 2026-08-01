"""
UAV multispectral image support.

Real multispectral sensors (e.g. MicaSense RedEdge/Altum, DJI P4 Multispectral)
capture separate band images (Blue, Green, Red, Red-edge, NIR), typically as
multi-band GeoTIFFs. When such a file is loaded, TRUE vegetation indices
(not RGB proxies) become possible.

HONESTY NOTE ON BAND ORDER: band order is sensor-specific and not
universally standardized. This loader assumes the common 5-band order
(Blue, Green, Red, Red-edge, NIR) used by MicaSense RedEdge-class sensors.
If your sensor uses a different order, the band mapping below must be
adjusted -- results will be silently wrong (not obviously so) if the
order doesn't match your actual sensor, so this assumption is stated
plainly rather than buried.
"""
from __future__ import annotations

import numpy as np

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

#: Default band order assumption for 5-band multispectral GeoTIFFs.
DEFAULT_BAND_ORDER = ["blue", "green", "red", "rededge", "nir"]


def is_multispectral_file(path: str) -> bool:
    """Quick check: does this GeoTIFF have more than 3 bands?"""
    if not RASTERIO_AVAILABLE:
        return False
    try:
        with rasterio.open(path) as src:
            return src.count > 3
    except Exception:
        return False


def load_multispectral(path: str, band_order: list[str] | None = None) -> dict:
    """
    Load a multi-band GeoTIFF and return a dict with 'rgb' (uint8 HxWx3 for
    display/alignment) and any of 'nir'/'rededge' bands found (float arrays,
    same value scale as the source file).

    Returns {"available": False, "note": ...} if rasterio isn't installed
    or the file can't be read as expected.
    """
    if not RASTERIO_AVAILABLE:
        return {"available": False,
                "note": "UNAVAILABLE: the 'rasterio' package is not installed. "
                        "Install with: pip install rasterio"}

    band_order = band_order or DEFAULT_BAND_ORDER
    try:
        with rasterio.open(path) as src:
            data = src.read()  # (bands, H, W)
            n_bands = data.shape[0]
            if n_bands < 3:
                return {"available": False, "note": f"File has only {n_bands} band(s); need at least 3 (RGB)."}

            band_map = {name: i for i, name in enumerate(band_order[:n_bands])}

            def norm(band: np.ndarray) -> np.ndarray:
                b = band.astype(np.float64)
                lo, hi = np.percentile(b, 1), np.percentile(b, 99)
                if hi <= lo:
                    return np.zeros_like(b, dtype=np.uint8)
                return np.clip((b - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

            rgb = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.uint8)
            for ch, name in enumerate(["red", "green", "blue"]):
                if name in band_map:
                    rgb[:, :, ch] = norm(data[band_map[name]])

            result = {"available": True, "rgb": rgb, "n_bands": n_bands, "band_order_used": band_order[:n_bands]}
            if "nir" in band_map:
                result["nir"] = data[band_map["nir"]].astype(np.float64)
            if "rededge" in band_map:
                result["rededge"] = data[band_map["rededge"]].astype(np.float64)
            return result
    except Exception as e:
        return {"available": False, "note": f"Could not read as multispectral GeoTIFF: {e}"}


def true_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Genuine NDVI from real Red and NIR bands (not a proxy)."""
    val = (nir - red) / (nir + red + 1e-6)
    return np.clip(val, -1.0, 1.0)


def true_ndre(rededge: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Genuine Normalized Difference Red-Edge Index from real bands (not a proxy)."""
    val = (nir - rededge) / (nir + rededge + 1e-6)
    return np.clip(val, -1.0, 1.0)


# ------------------------------------------------------------------ #
# Multi-file UAV bands.
#
# Many UAV multispectral sensors (e.g. MicaSense RedEdge/Altum) save each
# band as its OWN single-band file rather than one combined multi-band
# GeoTIFF. Selecting such files individually via a single-file loader
# means each one only "sees" 1 band and looks like an ordinary photo --
# which is exactly the bug this section fixes: multiple selected
# single-band files are grouped into ONE combined capture instead of
# being treated as unrelated images.
# ------------------------------------------------------------------ #

#: Filename substrings (lowercased) used to guess a band's role. Checked
#: in order; first match wins. Covers common MicaSense/DJI/generic naming.
BAND_KEYWORDS = [
    ("thermal", ["thermal", "lwir", "flir", "temp"]),
    ("rededge", ["rededge", "red_edge", "red-edge", "edge"]),
    ("nir", ["nir", "_5", "infrared"]),
    ("red", ["red", "_3"]),
    ("green", ["green", "_2"]),
    ("blue", ["blue", "_1"]),
]


def guess_band_role(filename: str) -> str:
    """Best-effort guess of which band a single-band file represents, from its name.
    Returns 'unknown' if nothing matches -- the user should confirm/correct via the
    band-assignment dialog rather than trust this blindly."""
    name = filename.lower()
    for role, keywords in BAND_KEYWORDS:
        if any(kw in name for kw in keywords):
            return role
    return "unknown"


def read_single_band(path: str) -> np.ndarray:
    """Read one single-band raster file as a 2D float array. Tries rasterio
    first (handles GeoTIFFs properly), falls back to Pillow for plain TIFFs."""
    if RASTERIO_AVAILABLE:
        try:
            with rasterio.open(path) as src:
                return src.read(1).astype(np.float64)
        except Exception:
            pass
    from PIL import Image
    img = Image.open(path)
    return np.array(img).astype(np.float64)


def build_capture_from_band_files(role_to_path: dict[str, str]) -> dict:
    """
    Combine separate single-band files (already assigned to roles, e.g.
    {'red': 'IMG_3.tif', 'nir': 'IMG_5.tif', ...}) into one capture, the
    same shape of result as `load_multiband_geotiff`: {'available', 'rgb',
    'nir'?, 'rededge'?, 'thermal'?}.
    """
    try:
        bands = {role: read_single_band(path) for role, path in role_to_path.items()}
    except Exception as e:
        return {"available": False, "note": f"Could not read one or more band files: {e}"}

    if "red" not in bands or "green" not in bands or "blue" not in bands:
        return {"available": False,
                "note": "Need at least Red, Green, and Blue band files assigned to build an RGB preview."}

    ref_shape = bands["red"].shape
    for role, arr in bands.items():
        if arr.shape != ref_shape:
            return {"available": False,
                    "note": f"Band '{role}' has shape {arr.shape}, expected {ref_shape} "
                            f"(all band files must be the same size)."}

    def norm(band: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(band, 1), np.percentile(band, 99)
        if hi <= lo:
            return np.zeros_like(band, dtype=np.uint8)
        return np.clip((band - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

    rgb = np.stack([norm(bands["red"]), norm(bands["green"]), norm(bands["blue"])], axis=-1)
    result = {"available": True, "rgb": rgb, "n_bands": len(bands), "band_order_used": list(bands.keys())}
    if "nir" in bands:
        result["nir"] = bands["nir"]
    if "rededge" in bands:
        result["rededge"] = bands["rededge"]
    if "thermal" in bands:
        result["thermal"] = bands["thermal"]
    return result


def custom_band_index(band_a: np.ndarray, band_b: np.ndarray) -> np.ndarray:
    """Generic Normalized Difference Index between any two bands: (A-B)/(A+B).
    Works for hyperspectral cubes or any two-band combination the user picks,
    without assuming specific named indices that require exact wavelength
    calibration this pipeline doesn't have."""
    a, b = band_a.astype(np.float64), band_b.astype(np.float64)
    val = (a - b) / (a + b + 1e-6)
    return np.clip(val, -1.0, 1.0)
