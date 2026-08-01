"""
Spectral reflectance proxy, band ratios, and pixel intensity.

HONESTY NOTE: without a radiometric calibration target (a reference panel
of known reflectance photographed under the same light) and known camera
response curves, raw pixel values cannot be converted to true physical
reflectance (0-1, unitless) or radiance (W/sr/m^2/nm). What is reported
below is a *relative brightness proxy* per channel (0-1, arbitrary units,
scene- and lighting-dependent) -- useful for comparing images captured
under similar conditions, not for absolute physical measurement.
"""
from __future__ import annotations

import numpy as np


def relative_reflectance_proxy(image_array: np.ndarray) -> dict:
    """
    Per-channel mean intensity, rescaled to [0, 1], reported as an
    UNCALIBRATED relative-brightness proxy for Blue/Green/Red. Red-edge and
    NIR are explicitly reported as unavailable.
    """
    r = float(image_array[:, :, 0].mean()) / 255.0
    g = float(image_array[:, :, 1].mean()) / 255.0
    b = float(image_array[:, :, 2].mean()) / 255.0
    return {
        "Blue": b, "Green": g, "Red": r,
        "Red-edge": None, "NIR": None,
        "note": "Uncalibrated relative brightness proxy (0-1), not true physical reflectance. "
                "Red-edge/NIR unavailable: RGB sensors do not capture these bands.",
    }


def spectral_radiance_status() -> dict:
    """Spectral radiance cannot be derived without a calibrated sensor and known exposure/illumination."""
    return {
        "available": False,
        "note": "UNAVAILABLE: radiance (W/sr/m^2/nm) requires a radiometrically calibrated sensor, known "
                "exposure settings, and a reference target. A standard photo cannot be converted to radiance.",
    }


def pixel_intensity(image_array: np.ndarray) -> dict:
    """Grayscale pixel intensity statistics -- genuinely computed, always available."""
    gray = np.array(image_array, dtype=float).mean(axis=2)
    return {"mean": float(gray.mean()), "std": float(gray.std()),
            "min": float(gray.min()), "max": float(gray.max())}


def band_ratios(image_array: np.ndarray) -> dict:
    """Simple real per-image band ratios computed from RGB means."""
    r = float(image_array[:, :, 0].mean())
    g = max(float(image_array[:, :, 1].mean()), 1.0)
    b = max(float(image_array[:, :, 2].mean()), 1.0)
    r_safe = max(r, 1.0)
    return {
        "Red/Green": r / g,
        "Red/Blue": r / b,
        "Green/Blue": (g) / b,
        "Green/Red (NIR-proxy/Red)": g / r_safe,
    }


def band_ratio_map(image_array: np.ndarray, numerator: str = "R", denominator: str = "G") -> np.ndarray:
    """Genuine per-pixel band-ratio raster, e.g. R/G, for map-canvas display.
    The denominator is floored to avoid extreme, meaningless ratios on
    near-black pixels (division by ~0)."""
    idx = {"R": 0, "G": 1, "B": 2}
    num = image_array[:, :, idx[numerator]].astype(float)
    den = np.maximum(image_array[:, :, idx[denominator]].astype(float), 8.0)
    return num / den
