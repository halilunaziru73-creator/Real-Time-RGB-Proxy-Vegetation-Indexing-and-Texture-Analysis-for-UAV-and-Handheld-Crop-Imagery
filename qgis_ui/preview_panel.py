"""
Central tab: Raw Preview.

Shows the raw photo for whichever loaded image is selected -- purely for
visual QC/reference. Loaded on demand (one image at a time, discarded
after), never stored, so this stays cheap even with thousands of images
registered. This is intentionally separate from the Map Canvas, which
shows only the aggregate composite's measured outputs.
"""
from __future__ import annotations

from PIL import Image
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel
from PyQt6.QtCore import Qt


class PreviewPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.paths: list[str] = []
        self.groups: dict[str, dict] = {}

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Preview image:"))
        self.selector = QComboBox()
        self.selector.setMinimumWidth(220)
        self.selector.currentIndexChanged.connect(self._show_selected)
        top.addWidget(self.selector)
        top.addStretch()
        layout.addLayout(top)

        self.image_label = QLabel("No images loaded yet.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("color: #666666;")
        layout.addWidget(self.image_label, stretch=1)

    def set_images(self, names: list[str], paths: list[str], groups: dict[str, dict] | None = None) -> None:
        self.paths = paths
        self.groups = groups or {}
        self.selector.blockSignals(True)
        self.selector.clear()
        self.selector.addItems(names)
        self.selector.blockSignals(False)
        if names:
            self.selector.setCurrentIndex(0)
            self._show_selected(0)

    def _show_selected(self, index: int) -> None:
        if index < 0 or index >= len(self.paths):
            return
        try:
            path = self.paths[index]
            if path in self.groups:
                img_array = self.groups[path]["rgb"]
                img = Image.fromarray(img_array)
            else:
                img = Image.open(path).convert("RGB")
            img.thumbnail((900, 700))
            qimage = QImage(img.tobytes(), img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qimage))
        except Exception as e:
            self.image_label.setText(f"Could not preview this image:\n{e}")
