"""
Left dock (secondary tab): Image Manager.

Lets the user point at a working directory (loading every image file
inside it, up to the 2000-image cap), and rename/delete/refresh entries
in the currently loaded image list -- without needing to re-browse files
one at a time.
"""
from __future__ import annotations

import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QFileDialog, QMessageBox, QInputDialog,
)
from PyQt6.QtCore import pyqtSignal

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")
MAX_IMAGES = 2000


class ImageManagerPanel(QWidget):
    """
    Shows the currently loaded image names. Supports:
      - Set Working Directory... (bulk-loads every image file in a folder)
      - Rename (double-click or button) -- renames the in-app label only,
        does NOT touch the file on disk, to avoid surprising data loss.
      - Delete Selected / Clear All -- removes images from the current
        session (also does not touch files on disk).
      - Refresh -- re-scans the working directory (if one was set) for
        added/removed files.
    """

    directoryLoaded = pyqtSignal(list)      # list[str] of paths
    imagesDeleted = pyqtSignal(list)        # list[int] indices removed
    imageRenamed = pyqtSignal(int, str)     # index, new display name
    refreshRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.working_directory: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Image Manager")
        header.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")
        layout.addWidget(header)

        self.dir_label = QLabel("No working directory set.")
        self.dir_label.setWordWrap(True)
        self.dir_label.setStyleSheet("color: #555555; font-size: 10px;")
        layout.addWidget(self.dir_label)

        dir_row = QHBoxLayout()
        self.set_dir_btn = QPushButton("Set Working Directory...")
        self.set_dir_btn.clicked.connect(self._choose_directory)
        dir_row.addWidget(self.set_dir_btn)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh)
        dir_row.addWidget(self.refresh_btn)
        layout.addLayout(dir_row)

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._rename_item)
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        self.rename_btn = QPushButton("Rename Selected")
        self.rename_btn.clicked.connect(self._rename_selected)
        btn_row.addWidget(self.rename_btn)
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self._delete_selected)
        btn_row.addWidget(self.delete_btn)
        layout.addLayout(btn_row)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self._clear_all)
        layout.addWidget(self.clear_btn)

    # ------------------------------------------------------------------ #
    def set_names(self, names: list[str]) -> None:
        self.list_widget.setUpdatesEnabled(False)
        try:
            self.list_widget.clear()
            for name in names:
                self.list_widget.addItem(QListWidgetItem(name))
        finally:
            self.list_widget.setUpdatesEnabled(True)

    # ------------------------------------------------------------------ #
    def _choose_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Working Directory")
        if not directory:
            return
        self.working_directory = directory
        self._scan_and_emit(directory)

    def _refresh(self) -> None:
        if self.working_directory:
            self._scan_and_emit(self.working_directory)
        else:
            self.refreshRequested.emit()

    def _scan_and_emit(self, directory: str) -> None:
        try:
            files = sorted(
                os.path.join(directory, f) for f in os.listdir(directory)
                if f.lower().endswith(IMAGE_EXTENSIONS)
            )
        except Exception as e:
            QMessageBox.critical(self, "Could not read directory", str(e))
            return

        if not files:
            QMessageBox.information(self, "No images found",
                                     f"No supported image files found in:\n{directory}")
            return

        truncated = len(files) > MAX_IMAGES
        if truncated:
            files = files[:MAX_IMAGES]
            QMessageBox.warning(self, "Too many images",
                                 f"Found more than {MAX_IMAGES} images; only the first {MAX_IMAGES} "
                                 f"(alphabetically) will be loaded.")

        self.dir_label.setText(f"Working directory: {directory}\n({len(files)} image(s) found)")
        self.directoryLoaded.emit(files)

    # ------------------------------------------------------------------ #
    def _rename_item(self, item: QListWidgetItem) -> None:
        self._rename(item)

    def _rename_selected(self) -> None:
        item = self.list_widget.currentItem()
        if item is None:
            QMessageBox.information(self, "No selection", "Select an image in the list first.")
            return
        self._rename(item)

    def _rename(self, item: QListWidgetItem) -> None:
        index = self.list_widget.row(item)
        new_name, ok = QInputDialog.getText(self, "Rename Image", "Display name:", text=item.text())
        if ok and new_name.strip():
            item.setText(new_name.strip())
            self.imageRenamed.emit(index, new_name.strip())

    def _delete_selected(self) -> None:
        rows = sorted((self.list_widget.row(i) for i in self.list_widget.selectedItems()), reverse=True)
        if not rows:
            QMessageBox.information(self, "No selection", "Select one or more images in the list first.")
            return
        for row in rows:
            self.list_widget.takeItem(row)
        self.imagesDeleted.emit(rows)

    def _clear_all(self) -> None:
        if self.list_widget.count() == 0:
            return
        confirm = QMessageBox.question(self, "Clear All Images",
                                        "Remove all loaded images from this session? (Files on disk are untouched.)")
        if confirm == QMessageBox.StandardButton.Yes:
            all_rows = list(range(self.list_widget.count()))
            self.list_widget.clear()
            self.imagesDeleted.emit(all_rows)
