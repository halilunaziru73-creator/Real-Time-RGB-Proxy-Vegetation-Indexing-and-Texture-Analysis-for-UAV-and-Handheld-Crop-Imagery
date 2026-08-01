# N_GACL: Real-Time RGB Proxy Vegetation Indexing and Texture Analysis for UAV and Handheld Crop Imagery

**Author:** Naziru Halilu

N_GACL is a desktop pipeline for exploratory agronomic image analysis, combining
two complementary components:

1. **The classical agronomic image-analysis pipeline** (`core/`, `qgis_ui/`,
   `main.py`, `main_classic.py`) — a desktop tool for exploratory agronomic image
   analysis, including streaming batch alignment, RGB-proxy vegetation indices,
   GLCM texture analysis, PCA/k-NN/K-Means clustering, optional pretrained
   ResNet101/Faster R-CNN models, and UAV multispectral support.

2. **The GACL reference implementation** (`gacl/`, `train_gacl.py`,
   `evaluate_gacl.py`, `check_learnable_structure.py`) — a runnable implementation
   of the four architectural components proposed in the accompanying paper
   (HGAViT, GCATT, DHGNN, VLAE) and their composite training objective.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the classical pipeline

```bash
python main.py            # QGIS-style GIS Workbench (PyQt6)
python main_classic.py    # simpler Tkinter app (no PyQt6 required)
```

## Running the GACL reference implementation

```bash
# Training run on the tabular reference dataset
python train_gacl.py --data GACL_Data.xlsx --epochs 5
python evaluate_gacl.py --data GACL_Data.xlsx --checkpoint gacl_smoketest_checkpoint.pt

# Dataset-validity check: assesses whether a given dataset supports a classifier
python check_learnable_structure.py --data GACL_Data.xlsx --target pathology

# Image-based training on real crop-disease photographs
python train_gacl_real_images.py --data_root /path/to/My_Data --epochs 20
```

`evaluate_gacl.py` computes accuracy, macro-precision/recall/F1, macro-AUC,
Cohen's kappa, and Matthews correlation coefficient from GACL's prototype-distance
classification head, alongside their chance-level reference values.

The GACL Measurements dock in the GUI (`qgis_ui/gacl_panel.py`) exposes the same
training and evaluation workflow directly in-app via "Select Real Image Folder..."
and "Train GACL on Real Images," printing the same metrics into the panel. This
runs synchronously in the current version; for larger datasets or longer runs,
using `train_gacl_real_images.py` directly from a terminal keeps the GUI
responsive.

## Directory layout

```
N_GACL/
├── main.py, main_classic.py       # classical pipeline entry points
├── core/                          # classical pipeline analysis modules
├── qgis_ui/                       # PyQt6 GIS Workbench interface (includes gacl_panel.py)
├── gacl/                          # GACL reference implementation
│   ├── hgavit.py                  # Hypergraph-guided attention vision transformer
│   ├── gcatt.py                   # Geometry-conditioned attention
│   ├── dhgnn.py                   # Dynamic hypergraph neural network
│   ├── vlae.py                    # Variational latent autoencoder
│   ├── losses.py                  # Composite training objective
│   ├── model.py                   # Composite model wiring
│   ├── dataset.py                 # Tabular dataset loader (GACL_Data.xlsx)
│   └── image_dataset.py           # Real image-based data loader
├── train_gacl.py, evaluate_gacl.py       # GACL training and evaluation scripts (tabular data)
├── train_gacl_real_images.py             # Image-based training and evaluation script
├── check_learnable_structure.py          # Dataset-validity check (tabular data)
├── GACL_Data.xlsx                        # 20,000-row tabular reference dataset
└── requirements.txt
```

## Data and evaluation notes

`GACL_Data.xlsx` contains 20,000 rows with a proper train/validation/test split
across 5 crops and 5 pathology classes. `check_learnable_structure.py` provides
an independent, model-agnostic check of whether a dataset carries exploitable
feature-label structure for a given target column, which is useful when curating
new datasets for training or evaluation.

## License

Released under the [MIT License](./LICENSE).
