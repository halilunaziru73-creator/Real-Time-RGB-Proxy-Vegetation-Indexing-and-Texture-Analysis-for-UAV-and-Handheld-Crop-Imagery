"""
Dialog: assign selected single-band UAV files to spectral roles.

Many UAV multispectral sensors save each band as its own file. This
dialog lets the user confirm (or correct) which file is which band --
auto-suggested from filenames, but always user-confirmed rather than
silently guessed, since a wrong guess would produce wrong (not obviously
wrong) index values.
"""
from __future__ import annotations

import os

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QComboBox, QPushButton, QLabel, QDialogButtonBox,
)

from core.multispectral import guess_band_role

ROLES = ["unknown", "red", "green", "blue", "rededge", "nir", "thermal", "ignore"]


class BandAssignmentDialog(QDialog):
    """Shows one row per selected file with a role dropdown, pre-filled by filename guess."""

    def __init__(self, file_paths: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign UAV Band Files")
        self.resize(560, 360)
        self.file_paths = file_paths

        layout = QVBoxLayout(self)
        info = QLabel(
            "Confirm which band each selected file represents. Roles were "
            "auto-guessed from filenames -- please check them before continuing, "
            "since a wrong assignment will produce wrong (not obviously wrong) index values."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.table = QTableWidget(len(file_paths), 2)
        self.table.setHorizontalHeaderLabels(["File", "Band Role"])
        self.combos: list[QComboBox] = []
        for row, path in enumerate(file_paths):
            name = os.path.basename(path)
            self.table.setItem(row, 0, QTableWidgetItem(name))
            combo = QComboBox()
            combo.addItems(ROLES)
            guess = guess_band_role(name)
            combo.setCurrentText(guess if guess in ROLES else "unknown")
            self.table.setCellWidget(row, 1, combo)
            self.combos.append(combo)
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def role_to_path(self) -> dict[str, str]:
        """Return {role: path} for every row not set to 'ignore'/'unknown'.
        If two rows share a role, the last one wins (dialog doesn't block that,
        but it's an edge case the user caused by mis-assigning)."""
        result = {}
        for path, combo in zip(self.file_paths, self.combos):
            role = combo.currentText()
            if role not in ("ignore", "unknown"):
                result[role] = path
        return result
