"""Bottom dock, tab 2: execution log."""
from __future__ import annotations

from datetime import datetime

from PyQt6.QtWidgets import QTextEdit


class LogPanel(QTextEdit):
    """Read-only, terminal-styled execution log."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.log("Application started.")

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.append(f"[{timestamp}] {message}")
