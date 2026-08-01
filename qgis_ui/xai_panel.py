"""Right dock widget: live descriptive statistics for the batch composite,
plus a cross-image similarity-based recommendation."""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit

from core.indices import image_stats, ndvi
from core.raster_layers import canopy_layer, susceptible_layer
from core import embedding_metrics


class XAIPanel(QWidget):
    """Shows real, freshly-computed statistics for the current aggregate
    composite, plus a real cross-image similarity recommendation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        header = QLabel("XAI & Recommendations")
        header.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")
        layout.addWidget(header)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet("background: white; color: #1f2d1e; font-family: Consolas, monospace;")
        layout.addWidget(self.text)

        self.clear()

    def clear(self) -> None:
        self.text.setPlainText(
            "Load images and click \u201cAlign & Build Composite\u201d to see live\n"
            "batch statistics and inspection priorities here."
        )

    def update_for_composite(self, composite, names: list[str], descriptors: np.ndarray | None) -> None:
        image_array = composite.mean_rgb
        stats = image_stats(image_array)
        veg = ndvi(image_array)
        mean_ndvi = float(np.nanmean(veg))
        canopy_pct = float(canopy_layer(image_array).mean() * 100)
        susceptible_pct = float(susceptible_layer(image_array).mean() * 100)
        edge_density = stats["edge_density"]
        variability = float(composite.std_map.mean())

        flagged = susceptible_pct > 15 or mean_ndvi < 0.2 or edge_density > 0.15
        priority = "HIGH — schedule field inspection" if flagged else "LOW — routine monitoring"

        lines = [
            f"Aggregate composite of {composite.count} aligned image(s)",
            "",
            "Descriptive statistics (computed on the composite):",
            f"  Mean intensity        : {stats['mean_intensity']:.2f}",
            f"  Intensity std dev     : {stats['std_intensity']:.2f}",
            f"  Edge density          : {edge_density:.4f}",
            f"  Mean NDVI             : {mean_ndvi:.3f}",
            f"  Canopy cover          : {canopy_pct:.1f}%",
            f"  Low-vigor / stressed  : {susceptible_pct:.1f}%",
            f"  Cross-batch variability: {variability:.2f} (higher = less consistent alignment/lighting)",
            "",
            "Rule-based inspection priority:",
            f"  {priority}",
        ]

        if descriptors is not None and len(names) >= 2:
            pw = embedding_metrics.pairwise_metrics(descriptors)
            sim_matrix = pw["cosine_similarity"]
            n = sim_matrix.shape[0]
            off_diag = sim_matrix[~np.eye(n, dtype=bool)]
            mean_sim = float(off_diag.mean())
            spread_sim = float(off_diag.std())

            if mean_sim > 0.9:
                cohesion_note = "Images are highly similar to each other overall -- the batch looks visually consistent."
            elif mean_sim > 0.6:
                cohesion_note = "Images show moderate similarity -- some visual variation across the batch."
            else:
                cohesion_note = "Images are quite visually diverse from each other -- expect varied conditions across the batch."

            lines.append("")
            lines.append("Cross-Image Similarity-Based Pathology Transfer Recommendation")
            lines.append("(real descriptor-similarity heuristic, not a trained cross-crop model):")
            lines.append(f"  Mean pairwise similarity : {mean_sim:.3f}")
            lines.append(f"  Similarity spread (std)  : {spread_sim:.3f}")
            lines.append(f"  {cohesion_note}")
            lines.append("  See the \u201cCross-Image Pathology Transfer Recommendation\u201d report "
                          "for the full similarity graph.")

        self.text.setPlainText("\n".join(lines))
