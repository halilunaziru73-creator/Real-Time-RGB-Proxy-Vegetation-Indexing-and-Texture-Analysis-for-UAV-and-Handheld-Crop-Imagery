"""
Central Map Canvas: shows ONLY the computed output for the current
AGGREGATE COMPOSITE (built by aligning and averaging every loaded image)
-- never a single raw photo, never a specific image's individual result.
Each enabled layer renders as its own correctly-labelled panel with a real
colorbar reflecting that index's true or declared value range.
"""
from __future__ import annotations

import math

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from core.multispectral import true_ndvi, true_ndre


class MapCanvas(QWidget):
    """Grid of correctly-scaled, labelled output panels for the batch composite."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(9, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self._array_cache: dict[str, np.ndarray] = {}
        self._composite_id: int | None = None
        self.clear()

    def clear(self) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5,
                 "Load images, click \u201cAlign && Build Composite\u201d, then tick a layer\n"
                 "in the Layers panel to see the batch's measured output here.",
                 ha="center", va="center", fontsize=11, color="#666666")
        self.canvas.draw_idle()

    def _cached_array(self, composite, key: str, array_func) -> np.ndarray:
        if id(composite) != self._composite_id:
            self._array_cache.clear()
            self._composite_id = id(composite)
        if key not in self._array_cache:
            self._array_cache[key] = array_func(composite.mean_rgb)
        return self._array_cache[key]

    def render(self, composite, raster_layers: list[dict], vector_layers: list[dict], title: str = "") -> None:
        panels = list(raster_layers) + [{"is_vector": True, **v} for v in vector_layers]
        self.figure.clear()

        if composite is None:
            self.clear()
            return
        if not panels:
            ax = self.figure.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.5, "Tick a layer in the Layers panel to see its\nmeasured output here.",
                    ha="center", va="center", fontsize=11, color="#666666")
            self.canvas.draw_idle()
            return

        n = len(panels)
        n_cols = min(3, n)
        n_rows = math.ceil(n / n_cols)
        self.figure.suptitle(title, fontsize=11)

        h, w = composite.mean_rgb.shape[0], composite.mean_rgb.shape[1]

        for i, layer in enumerate(panels):
            ax = self.figure.add_subplot(n_rows, n_cols, i + 1)

            if layer.get("is_vector"):
                color = layer["color"].name()
                ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], color=color, linewidth=1.5)
                ax.axvline(w / 2, color=color, linewidth=1.2)
                ax.axhline(h / 2, color=color, linewidth=1.2)
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)
                ax.set_title("Quadrant Boundaries (vector)", fontsize=9)
                ax.set_aspect("equal")
                continue

            array, note = self._resolve_layer_array(composite, layer)
            if array is None:
                ax.axis("off")
                ax.text(0.5, 0.5, note, ha="center", va="center", fontsize=8, wrap=True, color="#a33")
                ax.set_title(layer["name"], fontsize=9)
                continue

            vmin, vmax = layer["vmin"], layer["vmax"]
            if vmin is None or vmax is None:
                vmin, vmax = float(np.percentile(array, 2)), float(np.percentile(array, 98))
                if vmin == vmax:
                    vmin, vmax = float(array.min()), float(array.max() + 1e-6)

            im = ax.imshow(array, cmap=layer["cmap"], vmin=vmin, vmax=vmax)
            cbar = self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(layer["unit_label"], fontsize=8)
            ax.set_title(layer["name"], fontsize=9)
            ax.axis("off")

        self.figure.tight_layout(rect=[0, 0, 1, 0.95])
        self.canvas.draw_idle()

    def _resolve_layer_array(self, composite, layer: dict):
        """Handle the special composite-level layers (variability, true multispectral
        indices) that don't come from a simple per-pixel array_func on mean_rgb."""
        key = layer["key"]
        if key == "variability":
            return composite.std_map, None
        if key == "true_ndvi_ms":
            if composite.mean_nir is None:
                return None, ("No UAV multispectral NIR band available.\n"
                               "Load a multi-band UAV image to enable this layer.")
            red = composite.mean_rgb[:, :, 0].astype(float)
            return true_ndvi(red, composite.mean_nir), None
        if key == "true_ndre_ms":
            if composite.mean_nir is None or composite.mean_rededge is None:
                return None, ("No UAV multispectral red-edge/NIR bands available.\n"
                               "Load a multi-band UAV image to enable this layer.")
            return true_ndre(composite.mean_rededge, composite.mean_nir), None
        return self._cached_array(composite, key, layer["array_func"]), None

    def save(self, file_path: str) -> None:
        self.figure.savefig(file_path, dpi=200, bbox_inches="tight")
