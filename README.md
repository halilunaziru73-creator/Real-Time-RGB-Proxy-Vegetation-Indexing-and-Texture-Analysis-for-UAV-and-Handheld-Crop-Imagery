# N_GACL

N_GACL merges two previously separate codebases into one package:

1. **The classical agronomic image-analysis pipeline** (`core/`, `qgis_ui/`, `main.py`, `main_classic.py`) —
   a working, honesty-by-design desktop tool for exploratory agronomic image analysis
   (streaming batch alignment, RGB-proxy vegetation indices, GLCM texture, PCA/k-NN/K-Means,
   optional pretrained ResNet101/Faster R-CNN, UAV multispectral support). This part of the
   codebase is fully implemented and has been running since before this merge.

2. **The GACL reference implementation** (`gacl/`, `train_gacl.py`, `evaluate_gacl.py`,
   `check_learnable_structure.py`) — a real, runnable implementation of the four components
   proposed in the accompanying paper's Section 7 (HGAViT, GCATT, DHGNN, VLAE) and their
   composite training objective. This is new: a working encoder/loss/training-loop wiring
   exists where previously there was only a mathematical proposal.

## Honesty-by-design status of each part

| Component | Status |
|---|---|
| Classical pipeline (`core/`, `qgis_ui/`) | Implemented, real computation throughout, documented extensively in the paper (Sections 3–6, 8–9) |
| GACL code (`gacl/`) | Implemented and internally coherent (equation-to-code map in `gacl/README_GACL.md`); **not independently benchmarked as a validated classifier** |
| `GACL_Data.xlsx` | 20,000 rows, proper train/validation/test split, 5 crops × 5 pathology classes — **but empirically shown, by `check_learnable_structure.py`, to carry no exploitable feature–label structure for the `pathology` target** (RandomForest test accuracy 19.8–20.8% against a 20.0% chance level across every feature subset tried) |

**What this means concretely:** running `train_gacl.py` will execute the real GACL forward/backward pass and produce a genuine, decreasing loss curve — that confirms the *code* is correct. `evaluate_gacl.py` now additionally computes and prints real accuracy, macro-precision/recall/F1, macro-AUC, Cohen's kappa, and Matthews correlation coefficient from GACL's own prototype-distance classification head (Section 7.5) — genuine metrics, not proxies, printed alongside their chance-level reference value every time. Given `check_learnable_structure.py`'s finding above, the honest expectation is that these will land near chance level on the currently available data; that is GACL's real result on this file, not a script limitation, and should be reported exactly as printed rather than omitted or reframed.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the classical pipeline

```bash
python main.py            # QGIS-style GIS Workbench (PyQt6)
python main_classic.py    # simpler Tkinter app (no PyQt6 needed)
```

## Running the GACL reference implementation

```bash
# Smoke-test training run on the tabular stand-in dataset (prints loss components
# and, in evaluate_gacl.py, real accuracy/F1/AUC next to chance-level context)
python train_gacl.py --data GACL_Data.xlsx --epochs 5
python evaluate_gacl.py --data GACL_Data.xlsx --checkpoint gacl_smoketest_checkpoint.pt

# Independent empirical check of whether a given dataset supports any real classifier at all
python check_learnable_structure.py --data GACL_Data.xlsx --target pathology

# NEW: real image-based training, on real crop-disease photographs (gacl/image_dataset.py),
# once the classical-feature check (Section 6.9-6.12 of the paper) found real signal in
# this same dataset. Point --data_root at a folder-per-class image directory (see
# gacl/image_dataset.py's CLASS_TO_CROP_PATHOLOGY table for the exact folder names expected,
# or edit that table to match your own folder names).
python train_gacl_real_images.py --data_root /path/to/My_Data --epochs 20
```



**This is available directly from the GUI**: the GACL Measurements dock (`qgis_ui/gacl_panel.py`) has "Select Real Image Folder..." and "Train GACL on Real Images" buttons that run this same training/evaluation loop in-app, printing the same real, honest output (loss curve, then real accuracy/balanced-accuracy/macro-F1/kappa/MCC against chance-level context) directly into the panel. This runs synchronously and will block the GUI for the duration of training (a background-thread version would be a reasonable future improvement); for larger datasets or longer runs, using `train_gacl_real_images.py` directly from a terminal is faster and keeps the GUI responsive.


## Directory layout

```
N_GACL/
├── main.py, main_classic.py       # classical pipeline entry points
├── core/                            # classical pipeline analysis modules
├── qgis_ui/                          # PyQt6 GIS Workbench interface (includes gacl_panel.py)
├── gacl/                               # GACL reference implementation
│   ├── hgavit.py                         # Section 7.3
│   ├── gcatt.py                           # Section 7.5
│   ├── dhgnn.py                            # Section 7.6
│   ├── vlae.py                              # Section 7.7
│   ├── losses.py                             # Sections 7.4, 7.8
│   ├── model.py                                # composite wiring, Section 7.8
│   ├── dataset.py                                # tabular pseudo-patch loader (GACL_Data.xlsx)
│   └── image_dataset.py                          # NEW: real image-based loader (Section 6.9-6.12, 7.13)
├── train_gacl.py, evaluate_gacl.py    # GACL smoke-test scripts (tabular data)
├── train_gacl_real_images.py            # NEW: real image-based training/evaluation script
├── check_learnable_structure.py         # independent empirical validity check (tabular data)
├── GACL_Data.xlsx                          # 20,000-row tabular dataset (see status table above)
└── requirements.txt
```

