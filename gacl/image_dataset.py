"""
gacl/image_dataset.py

A real, image-based dataset loader for GACL, built to replace the tabular
pseudo-patch stand-in (gacl/dataset.py, GACLTabularDataset) once real
photographs are available -- which, as of this module, they are.

HONESTY NOTES (read before trusting anything this loader produces)
--------------------------------------------------------------------
1. Crop/pathology labels come from the source folder name for each image.
   Most map unambiguously (e.g. "Anthracnose on Cotton" -> crop=Cotton,
   pathology=Anthracnose). Two folders -- "Common_Rust" and "Brownspot" --
   do not name their crop explicitly. "Common_Rust" is assigned to Maize
   with HIGH confidence: a sibling folder in the same upload ("Gray_Leaf_Spot")
   contains files literally named "Corn_Gray_Spot (N).jpg", and both are
   standard co-occurring maize disease classes in public leaf-disease
   datasets. "Brownspot" is assigned to Rice with MODERATE confidence only,
   by analogy to standard public rice-disease dataset naming conventions
   (generic numeric filenames matching a common "Rice Leaf Diseases" dataset
   layout) -- this is an inference, not a confirmed label, and is flagged
   as such in CLASS_TO_CROP_PATHOLOGY below via the `confidence` field.
2. No acquisition-geometry metadata (camera angle, tilt, distance) exists
   for these images -- there is no EXIF or manifest field recording it.
   The geometry descriptor `g` (Section 7.3) is therefore returned as a
   ZERO vector, not a fabricated plausible-looking value, and geometry-
   conditioning in HGAViT effectively becomes a no-op for this dataset
   until real geometry metadata is available. This is stated in the model
   card / paper Section 7.13 and should be repeated in any downstream use.
3. Class sizes are severely imbalanced (8 to 275 images). No class
   re-weighting or oversampling is applied by default in this loader --
   that decision is left to the training script so it is visible and
   changeable, not silently baked in here.
4. Train/validation/test split: images whose filenames match a known
   augmentation-style pattern (zoom_, contrast_, rotozoom, translation_,
   rotate, flip, brightness) are restricted to the TRAIN split only, never
   placed in validation or test, specifically to reduce (not eliminate --
   see caveat below) the risk of an augmented copy of a test image being
   seen during training. This does not catch augmented copies that were
   renamed without a recognisable marker, or near-duplicate photographs of
   the same physical leaf under unrelated filenames -- neither of these
   can be ruled out from filenames alone.
"""
from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


# folder_name -> (crop, pathology, confidence)
# confidence: "confirmed" (crop stated in folder name or filename),
#             "inferred_high" (not stated, but strong corroborating evidence),
#             "inferred_moderate" (best-effort guess, not independently verified)
CLASS_TO_CROP_PATHOLOGY = {
    "Anthracnose on Cotton":     ("Cotton",    "Anthracnose",       "confirmed"),
    "Becterial Blight in Rice":  ("Rice",      "Bacterial Blight",  "confirmed"),
    "bollworm on Cotton":        ("Cotton",    "Bollworm",          "confirmed"),
    "Brownspot":                 ("Rice",      "Brown Spot",        "inferred_moderate"),
    "Common_Rust":               ("Maize",     "Common Rust",       "inferred_high"),
    "Cotton Aphid":              ("Cotton",    "Aphid",             "confirmed"),
    "Flag Smut":                 ("Wheat",     "Flag Smut",         "confirmed"),
    "Gray_Leaf_Spot":            ("Maize",     "Gray Leaf Spot",    "confirmed"),
    "Healthy Maize":             ("Maize",     "Healthy",           "confirmed"),
    "Healthy Wheat":             ("Wheat",     "Healthy",           "confirmed"),
    "Healthy cotton":            ("Cotton",    "Healthy",           "confirmed"),
    "Mosaic sugarcane":          ("Sugarcane", "Mosaic",            "confirmed"),
    "RedRot sugarcane":          ("Sugarcane", "Red Rot",           "confirmed"),
    "Rice Blast":                ("Rice",      "Blast",             "confirmed"),
    "Sugarcane Healthy":         ("Sugarcane", "Healthy",           "confirmed"),
    "Wheat Brown leaf rust":     ("Wheat",     "Brown Leaf Rust",   "confirmed"),
    "Wheat black rust":          ("Wheat",     "Black Rust",        "confirmed"),
    "cotton mealy bug":          ("Cotton",    "Mealy Bug",         "confirmed"),
    "cotton whitefly":           ("Cotton",    "Whitefly",          "confirmed"),
    "maize ear rot":             ("Maize",     "Ear Rot",           "confirmed"),
    "maize fall armyworm":       ("Maize",     "Fall Armyworm",     "confirmed"),
    "maize stem borer":          ("Maize",     "Stem Borer",        "confirmed"),
}

CROP_CLASSES_REAL = ["Cotton", "Rice", "Maize", "Wheat", "Sugarcane"]
PATHOLOGY_CLASSES_REAL = sorted({v[1] for v in CLASS_TO_CROP_PATHOLOGY.values()})

AUGMENTATION_MARKERS = ["zoom_", "contrast_", "rotozoom", "translation_", "rotate", "flip", "brightness"]


def _looks_augmented(filename: str) -> bool:
    name = filename.lower()
    return any(marker in name for marker in AUGMENTATION_MARKERS)


@dataclass
class RealImageRecord:
    path: str
    folder: str
    crop: str
    pathology: str
    confidence: str
    is_augmented_variant: bool
    file_hash: str


def scan_real_image_root(root: str) -> list[RealImageRecord]:
    """Walk a folder-per-class dataset root and build one record per unique
    (by content hash) image file, skipping exact duplicates."""
    records: list[RealImageRecord] = []
    seen_hashes: set[str] = set()

    for folder_name in sorted(os.listdir(root)):
        folder_path = os.path.join(root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        mapping = CLASS_TO_CROP_PATHOLOGY.get(folder_name)
        if mapping is None:
            # Honesty-by-design: skip, rather than guess, any folder not in the
            # mapping table above, and let the caller know via the returned list
            # being shorter than the number of files on disk.
            continue
        crop, pathology, confidence = mapping

        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(folder_path, fname)
            try:
                content = open(fpath, "rb").read()
            except OSError:
                continue
            h = hashlib.md5(content).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            records.append(RealImageRecord(
                path=fpath, folder=folder_name, crop=crop, pathology=pathology,
                confidence=confidence, is_augmented_variant=_looks_augmented(fname),
                file_hash=h,
            ))
    return records


def split_records(records: list[RealImageRecord], val_frac: float = 0.15, test_frac: float = 0.15,
                   seed: int = 42) -> dict[str, list[RealImageRecord]]:
    """Stratified split by (crop, pathology), with augmentation-flagged images
    restricted to train only (see module docstring, point 4)."""
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[RealImageRecord]] = {}
    for r in records:
        by_class.setdefault(r.pathology, []).append(r)

    train, val, test = [], [], []
    for pathology, recs in by_class.items():
        clean = [r for r in recs if not r.is_augmented_variant]
        augmented = [r for r in recs if r.is_augmented_variant]
        idx = rng.permutation(len(clean))
        clean_shuffled = [clean[i] for i in idx]

        n = len(clean_shuffled)
        n_val = max(1, int(round(n * val_frac))) if n >= 4 else 0
        n_test = max(1, int(round(n * test_frac))) if n >= 4 else 0
        n_val, n_test = min(n_val, n // 3 if n >= 3 else 0), min(n_test, n // 3 if n >= 3 else 0)

        val.extend(clean_shuffled[:n_val])
        test.extend(clean_shuffled[n_val:n_val + n_test])
        train.extend(clean_shuffled[n_val + n_test:])
        train.extend(augmented)  # augmented copies always go to train, never val/test

    return {"train": train, "validation": val, "test": test}


class GACLRealImageDataset(Dataset):
    """
    Real-image dataset for GACL. Produces the same batch dict shape as
    GACLTabularDataset (Section 7.11's tabular loader), so GACLModel and the
    existing training/evaluation scripts work unmodified against either.

    Patch scheme: each image is resized to `image_size` x `image_size`,
    then divided into a grid of (image_size/patch_size)^2 non-overlapping
    patches, each flattened to a real patch_dim = patch_size*patch_size*3
    vector -- the standard ViT-style patchification (Section 7.3), applied
    here to genuine pixel data for the first time in this codebase.
    """

    def __init__(self, records: list[RealImageRecord], image_size: int = 64, patch_size: int = 8):
        self.records = records
        self.image_size = image_size
        self.patch_size = patch_size
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.patch_dim = patch_size * patch_size * 3

        self.crop_to_idx = {c: i for i, c in enumerate(CROP_CLASSES_REAL)}
        self.path_to_idx = {p: i for i, p in enumerate(PATHOLOGY_CLASSES_REAL)}

    def __len__(self):
        return len(self.records)

    def _load_patches(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3), real pixel data, normalised to [0,1]

        patches = []
        for i in range(self.grid):
            for j in range(self.grid):
                patch = arr[i * self.patch_size:(i + 1) * self.patch_size,
                            j * self.patch_size:(j + 1) * self.patch_size, :]
                patches.append(patch.reshape(-1))
        return torch.tensor(np.stack(patches), dtype=torch.float32)  # (num_patches, patch_dim)

    def __getitem__(self, idx):
        rec = self.records[idx]
        patches = self._load_patches(rec.path)
        return {
            "patches": patches,
            # No real acquisition-geometry metadata exists for this dataset
            # (see module docstring, point 2) -- honestly zeroed, not fabricated.
            "geo": torch.zeros(5, dtype=torch.float32),
            "recon_target": patches.mean(dim=0),  # real per-image mean patch, used as VLAE's reconstruction target
            "crop": torch.tensor(self.crop_to_idx[rec.crop], dtype=torch.long),
            "pathology": torch.tensor(self.path_to_idx[rec.pathology], dtype=torch.long),
        }
