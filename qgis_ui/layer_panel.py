"""
Left dock widget: the Layer Controller / Table of Contents.

Each raster layer has a checkbox plus a Symbology dropdown listing a
curated set of scientifically-appropriate colormaps (diverging for signed
indices, sequential for masks/unbounded surfaces) -- real customization
without letting an arbitrary tint make a measured output unreadable.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QPushButton,
    QColorDialog, QLabel, QComboBox,
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, pyqtSignal

from core.raster_layers import RASTER_LAYER_DEFS, VECTOR_LAYER_DEFS, REPORT_LAYER_DEFS


class ColorSwatch(QPushButton):
    """Small clickable button showing/editing a vector layer's line colour."""

    colorChanged = pyqtSignal(QColor)

    def __init__(self, color_hex: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(22, 18)
        self._color = QColor(color_hex)
        self._refresh()
        self.clicked.connect(self._pick_color)

    def _refresh(self) -> None:
        self.setStyleSheet(f"background-color: {self._color.name()}; border: 1px solid #555;")

    def _pick_color(self) -> None:
        color = QColorDialog.getColor(self._color, self, "Choose layer colour")
        if color.isValid():
            self._color = color
            self._refresh()
            self.colorChanged.emit(color)

    def color(self) -> QColor:
        return self._color


class LayerPanel(QWidget):
    """Table-of-contents style layer controller with correctness-first symbology."""

    layersChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.raster_widgets: dict[str, dict] = {}
        self.vector_widgets: dict[str, dict] = {}
        self.report_checks: dict[str, QTreeWidgetItem] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Layers")
        header.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")
        layout.addWidget(header)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Layer", "Symbology"])
        self.tree.setColumnWidth(0, 220)
        self.tree.setColumnWidth(1, 110)
        self.tree.setAlternatingRowColors(True)
        layout.addWidget(self.tree)

        self._build_raster_group()
        self._build_vector_group()
        self._build_report_group()
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.expandAll()

    # ------------------------------------------------------------------ #
    def _bold_group(self, item: QTreeWidgetItem) -> None:
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)

    def _build_raster_group(self) -> None:
        group = QTreeWidgetItem(self.tree, ["Raster Layers (measured outputs)"])
        self._bold_group(group)

        for key, name, cmap_options, _fn, _vmin, _vmax, unit_label in RASTER_LAYER_DEFS:
            item = QTreeWidgetItem(group, [name])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Unchecked)
            item.setToolTip(0, unit_label)

            combo = QComboBox()
            combo.addItems(cmap_options)
            combo.currentIndexChanged.connect(lambda _i: self._emit_changed())
            self.tree.setItemWidget(item, 1, combo)

            self.raster_widgets[key] = {"item": item, "combo": combo}

    def _build_vector_group(self) -> None:
        group = QTreeWidgetItem(self.tree, ["Vector Layers"])
        self._bold_group(group)

        for key, name, color_hex in VECTOR_LAYER_DEFS:
            item = QTreeWidgetItem(group, [name])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Unchecked)

            swatch = ColorSwatch(color_hex)
            swatch.colorChanged.connect(lambda _c: self._emit_changed())
            self.tree.setItemWidget(item, 1, swatch)

            self.vector_widgets[key] = {"item": item, "swatch": swatch}

    def _build_report_group(self) -> None:
        group = QTreeWidgetItem(self.tree, ["Reports & Charts"])
        self._bold_group(group)

        for key, name in REPORT_LAYER_DEFS:
            item = QTreeWidgetItem(group, [name])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Unchecked)
            self.report_checks[key] = item

    # ------------------------------------------------------------------ #
    def _on_item_changed(self, _item: QTreeWidgetItem, _column: int) -> None:
        self._emit_changed()

    def _emit_changed(self) -> None:
        self.layersChanged.emit()

    # ------------------------------------------------------------------ #
    def active_raster_layers(self) -> list[dict]:
        """Return enabled raster layers with their (user-selectable) colormap/range/array-fn."""
        defs_by_key = {k: (name, fn, vmin, vmax, unit_label)
                       for k, name, _cmaps, fn, vmin, vmax, unit_label in RASTER_LAYER_DEFS}
        active = []
        for key, widgets in self.raster_widgets.items():
            item = widgets["item"]
            if item.checkState(0) == Qt.CheckState.Checked:
                name, fn, vmin, vmax, unit_label = defs_by_key[key]
                active.append({
                    "key": key, "name": name, "cmap": widgets["combo"].currentText(),
                    "array_func": fn, "vmin": vmin, "vmax": vmax, "unit_label": unit_label,
                })
        return active

    def active_vector_layers(self) -> list[dict]:
        active = []
        for key, widgets in self.vector_widgets.items():
            item = widgets["item"]
            if item.checkState(0) == Qt.CheckState.Checked:
                active.append({"key": key, "color": widgets["swatch"].color()})
        return active

    def checked_report_layers(self) -> list[str]:
        return [key for key, item in self.report_checks.items()
                if item.checkState(0) == Qt.CheckState.Checked]
