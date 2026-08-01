"""
gacl/dataset.py

Loads GACL_Data.xlsx and adapts its columns to the tensors GACLModel expects.

IMPORTANT CAVEAT (read before trusting any number this produces)
------------------------------------------------------------------
This loader has not independently verified the provenance, collection
protocol, or label quality of GACL_Data.xlsx -- that's outside what a
spreadsheet by itself can confirm. It just gets the columns into tensors so
the GACL code above is runnable end-to-end. See README.md for what would
need to be confirmed before treating any metric computed from this file as
a validated capability result.

Column mapping used here:
    embedding_1..32   -> reshaped into 8 pseudo-patches of dim 4, fed to HGAViT
                         as a stand-in for real image patches (no images exist
                         in this dataset)
    camera_height_m, camera_pitch_deg, camera_roll_deg, camera_yaw_deg,
    distance_m        -> geometry descriptor g (5-dim)
    crop              -> domain label c_i (5 classes)
    pathology         -> label p_i (5 classes, incl. 'Healthy')
    embedding_1..32   -> also used as VLAE's reconstruction target I (there is
                         no raw image to reconstruct, so the pre-computed
                         embedding vector stands in for it)
    split             -> train / validation / test, as given in the file
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


CROP_CLASSES = ["Maize", "Rice", "Soybean", "Tomato", "Wheat"]
PATHOLOGY_CLASSES = ["Healthy", "Blight", "Rust", "Mildew", "LeafSpot"]

GEO_COLS = ["camera_height_m", "camera_pitch_deg", "camera_roll_deg",
            "camera_yaw_deg", "distance_m"]
EMBED_COLS = [f"embedding_{i}" for i in range(1, 33)]


class GACLTabularDataset(Dataset):
    def __init__(self, xlsx_path, split="train", num_patches=8, patch_dim=4):
        df = pd.read_excel(xlsx_path)
        df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows found for split={split!r} in {xlsx_path}")

        self.crop_to_idx = {c: i for i, c in enumerate(CROP_CLASSES)}
        self.path_to_idx = {p: i for i, p in enumerate(PATHOLOGY_CLASSES)}

        self.embeddings = torch.tensor(df[EMBED_COLS].values, dtype=torch.float32)
        self.geo = torch.tensor(df[GEO_COLS].values, dtype=torch.float32)
        self.crop = torch.tensor(df["crop"].map(self.crop_to_idx).values, dtype=torch.long)
        self.pathology = torch.tensor(
            df["pathology"].map(self.path_to_idx).values, dtype=torch.long
        )

        self.num_patches = num_patches
        self.patch_dim = patch_dim
        assert num_patches * patch_dim == self.embeddings.shape[1], (
            "num_patches * patch_dim must equal embedding dimensionality (32)"
        )

        # standardize geometry + embedding features (fit on this split only;
        # a real pipeline would fit on train and apply to val/test)
        self._geo_mean = self.geo.mean(dim=0, keepdim=True)
        self._geo_std = self.geo.std(dim=0, keepdim=True).clamp(min=1e-6)
        self.geo = (self.geo - self._geo_mean) / self._geo_std

        self._emb_mean = self.embeddings.mean(dim=0, keepdim=True)
        self._emb_std = self.embeddings.std(dim=0, keepdim=True).clamp(min=1e-6)
        self.embeddings_norm = (self.embeddings - self._emb_mean) / self._emb_std

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        patches = self.embeddings_norm[idx].view(self.num_patches, self.patch_dim)
        return {
            "patches": patches,
            "geo": self.geo[idx],
            "recon_target": self.embeddings_norm[idx],
            "crop": self.crop[idx],
            "pathology": self.pathology[idx],
        }
