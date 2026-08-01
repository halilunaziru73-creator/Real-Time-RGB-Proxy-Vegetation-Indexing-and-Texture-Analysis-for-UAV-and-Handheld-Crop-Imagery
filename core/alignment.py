"""
Streaming batch alignment and aggregation.

Design goal: support up to ~2000 images WITHOUT holding them all in memory
at once. Each image is opened, resized to a common working size, aligned
to a reference via phase correlation, folded into running sum/sum-of-
squares accumulators, then discarded. Only a handful of small numbers
per image (its lightweight descriptor + QC stats) are kept afterwards.

HONESTY NOTE ON ALIGNMENT: this performs translation-only registration
(phase cross-correlation). It does NOT correct rotation, scale, or
perspective differences between photos. For photos taken from noticeably
different angles/distances, translation alignment will reduce but not
eliminate blur in the composite -- this is stated plainly in the UI
rather than implying a full geometric registration that isn't happening.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.registration import phase_cross_correlation
from skimage.transform import resize as sk_resize

from .indices import feature_vector, image_stats


@dataclass
class ImageRecord:
    name: str
    path: str
    orig_width: int
    orig_height: int
    mean_intensity: float
    shift_y: float
    shift_x: float
    included: bool
    error: str | None = None
    descriptor: np.ndarray | None = None
    has_multispectral: bool = False


@dataclass
class CompositeResult:
    mean_rgb: np.ndarray            # uint8 (H, W, 3) -- the aggregate composite
    std_map: np.ndarray             # float (H, W) -- per-pixel variability across the aligned stack
    count: int
    canvas_shape: tuple[int, int]
    mean_nir: np.ndarray | None = None
    mean_rededge: np.ndarray | None = None
    multispectral_count: int = 0


class BatchAligner:
    """
    Streaming aligner/accumulator. Call `add_image()` once per file, then
    `finalize()` to get the composite. Memory use is O(canvas size),
    independent of how many images were added.
    """

    def __init__(self, target_long_edge: int = 700, correlation_size: int = 256):
        self.target_long_edge = target_long_edge
        self.correlation_size = correlation_size

        self._ref_gray_small: np.ndarray | None = None
        self._canvas_shape: tuple[int, int] | None = None
        self._sum_rgb: np.ndarray | None = None
        self._sumsq_rgb: np.ndarray | None = None
        self._sum_nir: np.ndarray | None = None
        self._sum_rededge: np.ndarray | None = None
        self._count = 0
        self._ms_count = 0
        self.records: list[ImageRecord] = []

    # ------------------------------------------------------------------ #
    def _resize_to_canvas(self, arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape[:2]
        scale = self.target_long_edge / max(h, w)
        new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
        resized = sk_resize(arr, (new_h, new_w), preserve_range=True, anti_aliasing=True)
        return resized.astype(np.float64)

    def _fit_to_canvas_shape(self, arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Center-crop or edge-pad a (possibly differently-shaped) array to target_shape."""
        th, tw = target_shape
        h, w = arr.shape[:2]
        # Pad if smaller
        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)
        if pad_h or pad_w:
            pad_spec = [(pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)]
            if arr.ndim == 3:
                pad_spec.append((0, 0))
            arr = np.pad(arr, pad_spec, mode="edge")
            h, w = arr.shape[:2]
        # Crop if larger
        start_h = max(0, (h - th) // 2)
        start_w = max(0, (w - tw) // 2)
        arr = arr[start_h:start_h + th, start_w:start_w + tw, ...]
        return arr

    def add_image(self, name: str, path: str, pil_rgb: Image.Image,
                  extra_bands: dict[str, np.ndarray] | None = None) -> ImageRecord:
        """Process one image: resize, align, accumulate, then let it be garbage-collected."""
        try:
            orig_w, orig_h = pil_rgb.size
            arr = np.array(pil_rgb)
            resized = self._resize_to_canvas(arr)  # float64 (h, w, 3)

            gray_small = sk_resize(rgb2gray(resized / 255.0),
                                    (self.correlation_size, self.correlation_size),
                                    anti_aliasing=True)

            if self._ref_gray_small is None:
                self._ref_gray_small = gray_small
                self._canvas_shape = resized.shape[:2]
                shift_full = np.array([0.0, 0.0])
                aligned = resized
            else:
                shift, _error, _phase = phase_cross_correlation(
                    self._ref_gray_small, gray_small, upsample_factor=10)
                scale = resized.shape[0] / self.correlation_size
                shift_full = shift * scale
                aligned = ndimage.shift(resized, shift=(shift_full[0], shift_full[1], 0),
                                         order=1, mode="nearest")

            aligned = self._fit_to_canvas_shape(aligned, self._canvas_shape)

            if self._sum_rgb is None:
                self._sum_rgb = np.zeros(self._canvas_shape + (3,), dtype=np.float64)
                self._sumsq_rgb = np.zeros_like(self._sum_rgb)

            self._sum_rgb += aligned
            self._sumsq_rgb += aligned ** 2
            self._count += 1

            aligned_u8 = np.clip(aligned, 0, 255).astype(np.uint8)
            descriptor = feature_vector(aligned_u8)
            stats = image_stats(aligned_u8)

            has_ms = False
            if extra_bands:
                has_ms = True
                self._accumulate_extra_bands(extra_bands, shift_full)

            record = ImageRecord(
                name=name, path=path, orig_width=orig_w, orig_height=orig_h,
                mean_intensity=float(stats["mean_intensity"]),
                shift_y=float(shift_full[0]), shift_x=float(shift_full[1]),
                included=True, descriptor=descriptor, has_multispectral=has_ms,
            )
        except Exception as e:
            record = ImageRecord(
                name=name, path=path, orig_width=0, orig_height=0,
                mean_intensity=float("nan"), shift_y=float("nan"), shift_x=float("nan"),
                included=False, error=str(e),
            )
        self.records.append(record)
        return record

    def _accumulate_extra_bands(self, extra_bands: dict[str, np.ndarray], shift_full: np.ndarray) -> None:
        for key, arr in extra_bands.items():
            resized = sk_resize(arr.astype(np.float64), self._canvas_shape, preserve_range=True, anti_aliasing=True)
            aligned = ndimage.shift(resized, shift=(shift_full[0], shift_full[1]), order=1, mode="nearest")
            if key == "nir":
                if self._sum_nir is None:
                    self._sum_nir = np.zeros(self._canvas_shape, dtype=np.float64)
                self._sum_nir += aligned
            elif key == "rededge":
                if self._sum_rededge is None:
                    self._sum_rededge = np.zeros(self._canvas_shape, dtype=np.float64)
                self._sum_rededge += aligned
        self._ms_count += 1

    # ------------------------------------------------------------------ #
    def finalize(self) -> CompositeResult | None:
        if self._count == 0 or self._sum_rgb is None:
            return None
        mean_rgb = (self._sum_rgb / self._count)
        var_rgb = np.maximum(self._sumsq_rgb / self._count - mean_rgb ** 2, 0.0)
        std_map = np.sqrt(var_rgb).mean(axis=2)  # single 2D variability map, averaged over channels

        mean_nir = (self._sum_nir / self._ms_count) if self._sum_nir is not None and self._ms_count else None
        mean_rededge = (self._sum_rededge / self._ms_count) if self._sum_rededge is not None and self._ms_count else None

        return CompositeResult(
            mean_rgb=np.clip(mean_rgb, 0, 255).astype(np.uint8),
            std_map=std_map,
            count=self._count,
            canvas_shape=self._canvas_shape,
            mean_nir=mean_nir,
            mean_rededge=mean_rededge,
            multispectral_count=self._ms_count,
        )
