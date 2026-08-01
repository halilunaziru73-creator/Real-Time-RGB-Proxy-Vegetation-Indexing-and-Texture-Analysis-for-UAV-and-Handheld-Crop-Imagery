"""
Naziru Image Analysis Pipeline — entry point (QGIS-style GIS Workbench).

Run with:
    python main.py

This opens the PyQt6 desktop GUI styled after professional desktop-GIS
applications (QGIS-style layer panel, map canvas, XAI panel, attribute
table, and execution log).

Prefer the simpler, dependency-light Tkinter version instead? Run:
    python main_classic.py
"""
import sys

from PyQt6.QtWidgets import QApplication

from qgis_ui.main_window import QGISMainWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = QGISMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
