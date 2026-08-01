"""
Morphology, Environment & Relational Attribute Profile.

HONESTY NOTE: this module does NOT implement "Geometry-Agnostic
Contrastive Learning" or any trained neural encoder -- no model is
trained anywhere in this pipeline, and there is no labelled multi-crop
pathology dataset to train one on. What follows are genuine, classical
image-processing measurements organized under the same three conceptual
categories (pathological-morphology-like, geometric/environmental-like,
and cross-image relational), each computed directly from real pixels (or
real EXIF metadata where present) -- not learned, not simulated, and
never presented as the output of a trained contrastive model.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ExifTags
from skimage import filters, measure, morphology

from .indices import _gray, image_stats, ndvi
from .texture_features import glcm_features


# ------------------------------------------------------------------ #
# 1. Pathological-morphology-like proxies (real, classical CV)
# ------------------------------------------------------------------ #
def lesion_texture_pattern(image_array: np.ndarray) -> dict:
    """Real GLCM texture signature of the whole image (proxy for lesion micro-texture)."""
    feats = glcm_features(image_array)
    return {**feats, "note": "Real GLCM texture of the image. A true per-lesion texture "
                               "signature would need lesion segmentation with ground-truth masks."}


def necrotic_boundary_sharpness(image_array: np.ndarray) -> dict:
    """Real edge-gradient sharpness around the flagged low-vigor mask boundary."""
    veg = ndvi(image_array)
    threshold = np.nanpercentile(veg, 20)
    mask = veg <= threshold
    mask = morphology.remove_small_objects(mask, min_size=25)
    boundary = mask ^ morphology.binary_erosion(mask)
    gray = _gray(image_array)
    edges = filters.sobel(gray / 255.0)
    boundary_sharpness = float(edges[boundary].mean()) if boundary.any() else 0.0
    return {"boundary_sharpness": boundary_sharpness, "flagged_area_pct": float(mask.mean() * 100),
            "note": "Real Sobel-edge sharpness measured at the boundary of the heuristic low-NDVI mask."}


def chlorotic_discoloration(image_array: np.ndarray) -> dict:
    """Real yellowing/chlorosis colour proxy: (R+G-2B) normalized, over canopy pixels."""
    r = image_array[:, :, 0].astype(float)
    g = image_array[:, :, 1].astype(float)
    b = image_array[:, :, 2].astype(float)
    yellow_index = (r + g - 2 * b) / 255.0
    return {"mean_yellow_index": float(yellow_index.mean()), "max_yellow_index": float(yellow_index.max()),
            "note": "Real RGB-based yellowing proxy. Not a validated chlorophyll/chlorosis assay."}


def symptom_symmetry_distribution(image_array: np.ndarray) -> dict:
    """Real spatial-distribution stats of flagged low-vigor regions (centroid spread, region count)."""
    veg = ndvi(image_array)
    threshold = np.nanpercentile(veg, 20)
    mask = morphology.remove_small_objects(veg <= threshold, min_size=25)
    labeled = measure.label(mask)
    props = measure.regionprops(labeled)
    if not props:
        return {"region_count": 0, "note": "No flagged regions detected."}
    centroids = np.array([p.centroid for p in props])
    h, w = mask.shape
    normalized = centroids / np.array([h, w])
    spread = float(np.std(normalized, axis=0).mean())
    margin_bias = float(np.mean(np.minimum(normalized, 1 - normalized)))  # lower = closer to edges
    return {"region_count": len(props), "spatial_spread": spread, "margin_bias": margin_bias,
            "note": "Real spatial statistics of the heuristic low-vigor mask's connected regions."}


# ------------------------------------------------------------------ #
# 2. Geometric & environmental proxies (real: EXIF + colour/illumination stats)
# ------------------------------------------------------------------ #
def acquisition_geometry_from_exif(path: str) -> dict:
    """Real EXIF-derived acquisition metadata (camera tilt/GPS altitude), where present."""
    try:
        img = Image.open(path)
        exif_raw = img.getexif()
        if not exif_raw:
            return {"available": False, "note": "No EXIF metadata found in this file."}
        tags = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}
        result = {"available": True}
        if "GPSInfo" in tags:
            gps = tags["GPSInfo"]
            if isinstance(gps, dict) and 6 in gps:
                result["gps_altitude_m"] = float(gps[6])
        if "Orientation" in tags:
            result["orientation_tag"] = tags["Orientation"]
        if "FocalLength" in tags:
            result["focal_length_mm"] = float(tags["FocalLength"])
        if len(result) == 1:
            result["note"] = "EXIF present but no altitude/orientation/focal-length tags found."
        return result
    except Exception as e:
        return {"available": False, "note": f"Could not read EXIF: {e}"}


def illumination_context(image_array: np.ndarray) -> dict:
    """Real illumination diagnostics: brightness, shadow fraction, specular-highlight (glare) fraction."""
    gray = _gray(image_array)
    stats = image_stats(image_array)
    shadow_frac = float(np.mean(gray < 40))
    glare_frac = float(np.mean(gray > 245))
    return {
        "mean_brightness": stats["mean_intensity"],
        "shadow_fraction_pct": shadow_frac * 100,
        "glare_fraction_pct": glare_frac * 100,
        "note": "Real brightness/shadow/glare statistics from this image's own pixels.",
    }


def leaf_morphological_structure(image_array: np.ndarray) -> dict:
    """Real shape descriptors (aspect ratio, eccentricity, solidity) of the largest canopy region, as a rough
    leaf/canopy-shape proxy -- not a validated per-leaf morphology measurement."""
    r = image_array[:, :, 0].astype(float)
    g = image_array[:, :, 1].astype(float)
    b = image_array[:, :, 2].astype(float)
    total = r + g + b + 1e-6
    exg = 2 * (g / total) - (r / total) - (b / total)
    mask = morphology.remove_small_objects(exg > filters.threshold_otsu(exg), min_size=50)
    labeled = measure.label(mask)
    props = measure.regionprops(labeled)
    if not props:
        return {"available": False, "note": "No canopy region detected to profile."}
    largest = max(props, key=lambda p: p.area)
    return {
        "available": True,
        "aspect_ratio": float(largest.major_axis_length / (largest.minor_axis_length + 1e-6)),
        "eccentricity": float(largest.eccentricity),
        "solidity": float(largest.solidity),
        "note": "Real shape descriptors of the largest canopy region (ExG-segmented). Rough proxy, not "
                "per-leaf morphometrics (vein density/serration require dedicated leaf-scale imaging).",
    }


def background_noise_profile(image_array: np.ndarray) -> dict:
    """Real colour stats of the non-canopy (background/soil) region."""
    r = image_array[:, :, 0].astype(float)
    g = image_array[:, :, 1].astype(float)
    b = image_array[:, :, 2].astype(float)
    total = r + g + b + 1e-6
    exg = 2 * (g / total) - (r / total) - (b / total)
    canopy_mask = exg > filters.threshold_otsu(exg)
    bg_mask = ~canopy_mask
    if not bg_mask.any():
        return {"available": False, "note": "No background/soil pixels detected (canopy fills the frame)."}
    return {
        "available": True,
        "background_fraction_pct": float(bg_mask.mean() * 100),
        "background_mean_rgb": [float(r[bg_mask].mean()), float(g[bg_mask].mean()), float(b[bg_mask].mean())],
        "note": "Real colour statistics of the non-canopy region (ExG-segmented background/soil).",
    }


def morphology_environment_profile(image_array: np.ndarray, path: str | None = None) -> dict:
    """Bundle all category-1 and category-2 proxies for one image."""
    return {
        "pathological_morphology_proxies": {
            "lesion_texture_pattern": lesion_texture_pattern(image_array),
            "necrotic_boundary_sharpness": necrotic_boundary_sharpness(image_array),
            "chlorotic_discoloration": chlorotic_discoloration(image_array),
            "symptom_symmetry_distribution": symptom_symmetry_distribution(image_array),
        },
        "geometric_environmental_proxies": {
            "acquisition_geometry_exif": acquisition_geometry_from_exif(path) if path else
                {"available": False, "note": "No file path provided."},
            "illumination_context": illumination_context(image_array),
            "leaf_morphological_structure": leaf_morphological_structure(image_array),
            "background_noise_profile": background_noise_profile(image_array),
        },
    }
