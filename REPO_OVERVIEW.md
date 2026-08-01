# Repository Overview

This repository brings together the components of the **N_GACL** project:

- **Source code** (`core/`, `qgis_ui/`, `gacl/`, `main.py`, `main_classic.py`, `train_gacl.py`, etc.)
  — the classical agronomic image-analysis pipeline plus the GACL reference implementation.
  See [`README.md`](README.md) for the original project README.
- **`paper/`** — the accompanying manuscript, *"Real Time RGB Proxy Vegetation Indexing and
  Texture Analysis for UAV and Handheld Crop Imagery"* (Naziru Halilu, Juwairiyyah Sulaiman).
- **`data_sample/`** — a small representative sample of the crop-disease image dataset used
  for validation (see `data_sample/README.md` for details; the full dataset is not included).
- **`build/`** — PyInstaller build output for the packaged desktop application. These are
  large compiled binaries tracked via **Git LFS** (see `.gitattributes`).
- **`dist/main.exe`** — the packaged Windows executable, also tracked via Git LFS.
- **`GACL_Data.xlsx`** — the 20,000-row reference dataset used as a negative control
  (also tracked via Git LFS due to size).

## Note on large files (Git LFS)

Several files in this repository exceed GitHub's normal size limits and are tracked with
[Git LFS](https://git-lfs.github.com/):

- `build/main/main.pkg`, `build/main/PYZ-00.pyz`
- `dist/main.exe`
- `GACL_Data.xlsx`

If you clone this repo, install Git LFS first (`git lfs install`) so these files download
correctly instead of showing up as small pointer files.
