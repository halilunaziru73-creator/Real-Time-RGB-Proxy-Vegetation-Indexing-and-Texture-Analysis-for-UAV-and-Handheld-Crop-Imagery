"""Bottom dock, tab 1: attribute data table (one row per loaded image).

Populated from the lightweight per-image records produced during
alignment (core.alignment.ImageRecord) -- not from full images kept in
memory, so this scales to large batches.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem
from PyQt6.QtCore import Qt

COLUMNS = [
    "Image", "Width", "Height", "Mean Intensity", "Mean NDVI (proxy)",
    "Align Shift Y", "Align Shift X", "Included", "Note",
]

#: Above this row count, skip the expensive resizeColumnsToContents() pass
#: (O(rows x cols) content measurement) and use fixed widths instead, so
#: large batches (hundreds/thousands of images) don't freeze the UI.
FAST_MODE_ROW_THRESHOLD = 200
DEFAULT_COLUMN_WIDTHS = [220, 70, 70, 110, 130, 100, 100, 70, 220]


class AttributeTable(QTableWidget):
    """QGIS-style attribute table: one row per image, one column per statistic."""

    def __init__(self, parent=None):
        super().__init__(0, len(COLUMNS), parent)
        self.setHorizontalHeaderLabels(COLUMNS)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.horizontalHeader().setStretchLastSection(True)

    def _apply_widths_fast(self) -> None:
        for col, width in enumerate(DEFAULT_COLUMN_WIDTHS):
            self.setColumnWidth(col, width)

    def populate_from_records(self, records: list) -> None:
        self.setUpdatesEnabled(False)
        try:
            self.setRowCount(len(records))
            for row, rec in enumerate(records):
                mean_ndvi = f"{rec.descriptor[6]:.3f}" if rec.descriptor is not None else "N/A"
                values = [
                    rec.name, str(rec.orig_width), str(rec.orig_height),
                    f"{rec.mean_intensity:.2f}" if rec.included else "N/A",
                    mean_ndvi,
                    f"{rec.shift_y:.2f}" if rec.included else "N/A",
                    f"{rec.shift_x:.2f}" if rec.included else "N/A",
                    "Yes" if rec.included else "No",
                    rec.error or "",
                ]
                for col, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.setItem(row, col, item)
            if len(records) <= FAST_MODE_ROW_THRESHOLD:
                self.resizeColumnsToContents()
            else:
                self._apply_widths_fast()
        finally:
            self.setUpdatesEnabled(True)

    def populate_names_only(self, names: list[str]) -> None:
        """Lightweight placeholder rows before alignment has been run."""
        self.setUpdatesEnabled(False)
        try:
            self.setRowCount(len(names))
            for row, name in enumerate(names):
                item = QTableWidgetItem(name)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.setItem(row, 0, item)
                for col in range(1, len(COLUMNS)):
                    placeholder = "(run Align & Build Composite)" if col == len(COLUMNS) - 1 else ""
                    self.setItem(row, col, QTableWidgetItem(placeholder))
            if len(names) <= FAST_MODE_ROW_THRESHOLD:
                self.resizeColumnsToContents()
            else:
                self._apply_widths_fast()
        finally:
            self.setUpdatesEnabled(True)
