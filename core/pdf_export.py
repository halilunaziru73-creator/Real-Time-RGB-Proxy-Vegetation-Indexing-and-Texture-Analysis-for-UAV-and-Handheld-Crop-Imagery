"""Combine one or more matplotlib Figures into a single multi-page PDF report."""
from __future__ import annotations

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


def export_figures_to_pdf(path: str, figures: list[Figure], titles: list[str] | None = None) -> None:
    """Write each figure as its own page in one PDF file at `path`."""
    with PdfPages(path) as pdf:
        for i, fig in enumerate(figures):
            if titles and i < len(titles):
                fig.suptitle(titles[i], fontsize=12)
            pdf.savefig(fig, bbox_inches="tight")
