"""
Bottom dock, tab 3: Field & Sensor Data.

Everything in this table is USER-SUPPLIED. Nothing here is computed or
guessed by the pipeline -- these are exactly the metrics (plant height,
LAI, soil/weather readings, yield, disease scores, etc.) that genuinely
require instruments or ground-truth observation, not image analysis.

PERFORMANCE NOTE: building this table eagerly on every image load used to
create (rows x 26) table-cell widgets synchronously -- for large batches
(hundreds/thousands of images) that froze the whole window for a long
time, which could look like the UI "losing" its docks. Table rows are now
built only when explicitly requested (button), and skip the expensive
existing-value migration pass above a size threshold.
"""
from __future__ import annotations

import csv

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QFileDialog, QMessageBox,
)

from core.field_data import FIELD_METRIC_LABELS, FIELD_METRIC_KEYS

MIGRATE_VALUES_MAX_ROWS = 300  # above this, skip the "preserve existing entries" pass for speed


class FieldDataPanel(QWidget):
    """Editable table of user-supplied field/sensor measurements, one row per image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._y_true: list | None = None
        self._y_pred: list | None = None
        self._pending_names: list[str] = []
        self._row_names: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        info = QLabel(
            "All values below are entered by you -- none are computed or guessed from the "
            "photo. Fill in only what you actually measured (soil probe, weather station, "
            "ruler, lab assay, ground count, etc.)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #555555; padding: 4px;")
        layout.addWidget(info)

        button_row = QHBoxLayout()
        self.build_table_btn = QPushButton("Prepare Table for Loaded Images")
        self.build_table_btn.clicked.connect(self._build_table_now)
        button_row.addWidget(self.build_table_btn)
        self.load_labels_btn = QPushButton("Load Labels CSV for Classification Metrics...")
        self.load_labels_btn.clicked.connect(self._load_labels_csv)
        button_row.addWidget(self.load_labels_btn)
        self.labels_status = QLabel("No label CSV loaded.")
        self.labels_status.setStyleSheet("color: #555555;")
        button_row.addWidget(self.labels_status)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.status_label = QLabel("No images registered yet.")
        self.status_label.setStyleSheet("color: #555555; font-style: italic;")
        layout.addWidget(self.status_label)

        self.table = QTableWidget(0, len(FIELD_METRIC_LABELS))
        self.table.setHorizontalHeaderLabels(FIELD_METRIC_LABELS)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

    # ------------------------------------------------------------------ #
    def populate(self, image_names: list[str]) -> None:
        """
        Record which images are registered, but do NOT build table rows yet
        -- that happens on demand via the "Prepare Table" button, so loading
        hundreds/thousands of images stays fast.
        """
        self._pending_names = list(image_names)
        if image_names:
            self.status_label.setText(
                f"{len(image_names)} image(s) registered. Click \u201cPrepare Table for Loaded "
                f"Images\u201d to enter field/sensor data for them."
            )
        else:
            self.status_label.setText("No images registered yet.")

    def _build_table_now(self) -> None:
        image_names = self._pending_names
        if not image_names:
            QMessageBox.information(self, "No images", "Load some images first.")
            return

        existing = {}
        if self.table.rowCount() and self._row_names and len(self._row_names) <= MIGRATE_VALUES_MAX_ROWS:
            for row, name in enumerate(self._row_names):
                existing[name] = [self.table.item(row, c).text() if self.table.item(row, c) else ""
                                   for c in range(self.table.columnCount())]

        self.table.setRowCount(0)  # clear fast before rebuilding
        self.table.setRowCount(len(image_names))
        self._row_names = list(image_names)
        for row, name in enumerate(image_names):
            values = existing.get(name, ["" for _ in FIELD_METRIC_LABELS])
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))

        self.status_label.setText(f"Table ready for {len(image_names)} image(s).")

    def get_group_labels(self) -> list[str] | None:
        """Values from the 'Group / Crop Label' column, or None if not usable."""
        try:
            col = FIELD_METRIC_KEYS.index("group_label")
        except ValueError:
            return None
        if not self._row_names:
            return None
        entered = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, col)
            entered.append(item.text().strip() if item and item.text().strip() else None)
        if len({v for v in entered if v}) < 2:
            return None
        return [v if v else f"unlabeled_{i}" for i, v in enumerate(entered)]

    def get_group_labels_for_names(self, names: list[str]) -> list[str] | None:
        """
        Same as get_group_labels(), but returned in the order of (and only
        for) the given names -- used because after alignment, some
        registered images may have been skipped/failed, so the descriptor
        list can be a subset of all registered rows. Returns None if the
        table hasn't been built yet, or fewer than 2 real labels were
        actually entered (placeholder "unlabeled" rows never count as
        real distinct classes).
        """
        try:
            col = FIELD_METRIC_KEYS.index("group_label")
        except ValueError:
            return None
        if not self._row_names:
            return None
        label_by_name = {}
        for row, row_name in enumerate(self._row_names):
            item = self.table.item(row, col)
            label_by_name[row_name] = item.text().strip() if item and item.text().strip() else None

        entered = [label_by_name.get(n) for n in names]
        if len({v for v in entered if v}) < 2:
            return None
        return [v if v else f"unlabeled_{i}" for i, v in enumerate(entered)]

    def get_classification_labels(self):
        return self._y_true, self._y_pred

    # ------------------------------------------------------------------ #
    def _load_labels_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Labels CSV (columns: y_true, y_pred)", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "y_true" not in reader.fieldnames or "y_pred" not in reader.fieldnames:
                    raise ValueError("CSV must have 'y_true' and 'y_pred' columns.")
                y_true, y_pred = [], []
                for row in reader:
                    y_true.append(row["y_true"])
                    y_pred.append(row["y_pred"])
            self._y_true, self._y_pred = y_true, y_pred
            self.labels_status.setText(f"Loaded {len(y_true)} labelled row(s) from {path.split('/')[-1]}")
        except Exception as e:
            QMessageBox.critical(self, "Could not load labels", str(e))
            self.labels_status.setText("No label CSV loaded.")
