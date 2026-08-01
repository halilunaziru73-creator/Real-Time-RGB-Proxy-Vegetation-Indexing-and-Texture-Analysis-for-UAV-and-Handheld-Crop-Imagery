"""
Naziru Image Analysis Pipeline — classic Tkinter entry point.

Run with:
    python main_classic.py

This is the simpler, dependency-light version (no PyQt6 required). Click
"Browse Images" to select one or more agronomic images (the INPUT), tick
the analyses you want, then click "Run Analysis" to compute and save the
results (the OUTPUT), which are also previewed inline in the app.

For the QGIS-style GIS Workbench (layer panel, symbology, map canvas,
XAI panel, attribute table), run `python main.py` instead.
"""
import tkinter as tk

from core.gui import AgronomicGUI


def main() -> None:
    root = tk.Tk()
    AgronomicGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
