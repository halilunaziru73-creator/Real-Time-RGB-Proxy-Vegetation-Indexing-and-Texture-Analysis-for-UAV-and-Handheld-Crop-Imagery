"""
GACL Measurements dock panel.

Shows real output from the GACL reference implementation (gacl/) when it is
runnable in the current environment, and an honest, specific "unavailable"
message otherwise -- the same honesty-by-design rule used by every other
panel in this application (Section 8 of the accompanying paper).

Three distinct states are shown, never blurred together:
  1. torch not installed             -> explicit install instruction, no numbers shown.
  2. torch installed, no checkpoint   -> explicit instruction to run train_gacl.py first.
  3. checkpoint present               -> runs evaluate_gacl.py's real metrics computation
                                          and displays it WITH chance-level context,
                                          exactly as evaluate_gacl.py itself prints it.

This panel never prints a bare accuracy/F1/AUC number without the chance-level
reference alongside it, and it never fabricates a result when the model or
data are unavailable.
"""
from __future__ import annotations

import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton,
    QFileDialog, QMessageBox,
)


class GACLPanel(QWidget):
    """Dock widget showing GACL (Section 7) measurements, honestly scoped."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_path: str | None = None
        self.checkpoint_path: str | None = None
        self.image_root: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        header = QLabel("GACL Measurements (Section 7)")
        header.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")
        layout.addWidget(header)

        subheader = QLabel(
            "Geometry-Agnostic Contrastive Learning is a proposed, code-implemented\n"
            "architecture. Metrics below are real when shown, always printed next to\n"
            "their chance-level reference, never fabricated when unavailable."
        )
        subheader.setStyleSheet("color: #555; font-size: 10px; padding: 2px 4px;")
        subheader.setWordWrap(True)
        layout.addWidget(subheader)

        btn_row = QHBoxLayout()
        self.select_data_btn = QPushButton("Select GACL_Data.xlsx...")
        self.select_data_btn.clicked.connect(self._select_data)
        btn_row.addWidget(self.select_data_btn)

        self.select_ckpt_btn = QPushButton("Select Checkpoint...")
        self.select_ckpt_btn.clicked.connect(self._select_checkpoint)
        btn_row.addWidget(self.select_ckpt_btn)
        layout.addLayout(btn_row)

        self.run_check_btn = QPushButton("Run Learnable-Structure Check (RandomForest)")
        self.run_check_btn.clicked.connect(self._run_learnable_structure_check)
        layout.addWidget(self.run_check_btn)

        self.run_eval_btn = QPushButton("Run GACL Evaluation (real accuracy/F1/AUC)")
        self.run_eval_btn.clicked.connect(self._run_gacl_evaluation)
        layout.addWidget(self.run_eval_btn)

        train_row = QHBoxLayout()
        self.select_image_root_btn = QPushButton("Select Real Image Folder...")
        self.select_image_root_btn.clicked.connect(self._select_image_root)
        train_row.addWidget(self.select_image_root_btn)

        self.train_gacl_btn = QPushButton("Train GACL on Real Images")
        self.train_gacl_btn.clicked.connect(self._train_gacl_on_real_images)
        train_row.addWidget(self.train_gacl_btn)
        layout.addLayout(train_row)

        self.epochs_label = QLabel("Epochs: 20 (edit in code / rerun with a different value if needed)")
        self.epochs_label.setStyleSheet("color: #555; font-size: 9px;")
        layout.addWidget(self.epochs_label)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet("background: white; color: #1f2d1e; font-family: Consolas, monospace;")
        layout.addWidget(self.text)

        self._check_environment()

    # ------------------------------------------------------------------ #
    def _check_environment(self) -> None:
        try:
            import torch  # noqa: F401
            self._torch_available = True
        except ImportError:
            self._torch_available = False

        if not self._torch_available:
            self.text.setPlainText(
                "GACL evaluation UNAVAILABLE in this environment.\n\n"
                "Reason: the 'torch' package is not installed.\n"
                "Install with: pip install torch\n\n"
                "No GACL metrics are shown until this is resolved -- this panel\n"
                "does not substitute a placeholder number for a real one."
            )
            self.run_check_btn.setEnabled(True)   # RandomForest check only needs pandas/sklearn
            self.run_eval_btn.setEnabled(False)
        else:
            self.text.setPlainText(
                "torch is available. Select GACL_Data.xlsx to run the learnable-\n"
                "structure check, and a checkpoint (from train_gacl.py) to run real\n"
                "GACL evaluation metrics."
            )

    def _select_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select GACL_Data.xlsx", "", "Excel files (*.xlsx)")
        if path:
            self.data_path = path
            self.text.append(f"\nData file selected: {path}")

    def _select_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select GACL checkpoint", "", "PyTorch checkpoint (*.pt)")
        if path:
            self.checkpoint_path = path
            self.text.append(f"\nCheckpoint selected: {path}")

    # ------------------------------------------------------------------ #
    def _run_learnable_structure_check(self) -> None:
        if not self.data_path or not os.path.exists(self.data_path):
            QMessageBox.warning(self, "No data file", "Select GACL_Data.xlsx first.")
            return
        try:
            import pandas as pd
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.dummy import DummyClassifier
            from sklearn.metrics import accuracy_score, f1_score
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            self.text.setPlainText(f"Learnable-structure check UNAVAILABLE: {e}\nInstall with: pip install pandas scikit-learn openpyxl")
            return

        try:
            df = pd.read_excel(self.data_path)
            train_df = df[df["split"] == "train"]
            test_df = df[df["split"] == "test"]
            y_train, y_test = train_df["pathology"], test_df["pathology"]

            embed_cols = [c for c in df.columns if c.startswith("embedding_")]
            latent_cols = [c for c in df.columns if c.startswith("latent_z")]
            geo_cols = [c for c in ["camera_height_m", "camera_pitch_deg", "camera_roll_deg",
                                     "camera_yaw_deg", "distance_m"] if c in df.columns]
            raw_cols = [c for c in ["mean_R", "mean_G", "mean_B", "ExG", "VARI",
                                     "GLCM_contrast", "GLCM_entropy", "area", "perimeter", "solidity"]
                        if c in df.columns]

            feature_sets = {
                "embeddings_only": embed_cols,
                "raw_visual_features_only": raw_cols,
                "embeddings+latents+geo": embed_cols + latent_cols + geo_cols,
                "all_numeric_features": embed_cols + latent_cols + geo_cols + raw_cols,
            }

            n_classes = y_test.nunique()
            chance = 1.0 / n_classes
            lines = [
                "=" * 60,
                "Empirical learnable-structure check (real RandomForest, run now)",
                "=" * 60,
                f"Train rows: {len(train_df)}   Test rows: {len(test_df)}",
                f"Chance level for {n_classes} balanced classes: {chance:.4f}",
                "",
            ]
            for name, cols in feature_sets.items():
                cols = [c for c in cols if c in df.columns]
                if not cols:
                    continue
                scaler = StandardScaler().fit(train_df[cols].values)
                X_train = scaler.transform(train_df[cols].values)
                X_test = scaler.transform(test_df[cols].values)
                clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
                acc = accuracy_score(y_test, pred)
                f1 = f1_score(y_test, pred, average="macro")
                lines.append(f"[{name}] accuracy={acc:.4f}  macro-F1={f1:.4f}")

            dummy = DummyClassifier(strategy="stratified", random_state=42)
            dummy.fit(train_df[embed_cols], y_train)
            pred_dummy = dummy.predict(test_df[embed_cols])
            lines.append("")
            lines.append(f"[stratified-random baseline] accuracy={accuracy_score(y_test, pred_dummy):.4f}")
            lines.append("")
            lines.append("Read this as: if every feature set sits within noise of the")
            lines.append("baseline and chance level, no usable signal was found for this")
            lines.append("target in this file. This is real, freshly computed just now.")

            self.text.setPlainText("\n".join(lines))
        except Exception as e:
            self.text.setPlainText(f"Check failed: {e}")

    def _run_gacl_evaluation(self) -> None:
        if not self._torch_available:
            QMessageBox.warning(self, "torch unavailable", "Install torch first (see panel text).")
            return
        if not self.data_path or not self.checkpoint_path:
            QMessageBox.warning(self, "Missing input", "Select both GACL_Data.xlsx and a checkpoint first.")
            return
        try:
            import numpy as np
            import torch
            from torch.utils.data import DataLoader
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                roc_auc_score, cohen_kappa_score, matthews_corrcoef,
            )
            from gacl import GACLConfig, GACLModel, GACLTabularDataset, CROP_CLASSES, PATHOLOGY_CLASSES
        except ImportError as e:
            self.text.setPlainText(f"GACL evaluation UNAVAILABLE: {e}")
            return

        try:
            cfg = GACLConfig()
            device = torch.device("cpu")
            test_ds = GACLTabularDataset(self.data_path, split="test",
                                          num_patches=cfg.hgavit.num_patches,
                                          patch_dim=cfg.hgavit.patch_dim)
            loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            n_classes = len(PATHOLOGY_CLASSES)
            chance = 1.0 / n_classes

            model = GACLModel(cfg, num_crops=len(CROP_CLASSES), recon_input_dim=cfg.vlae.recon_dim).to(device)
            model.load_state_dict(torch.load(self.checkpoint_path, map_location=device))
            model.eval()

            all_probs, all_preds, all_true = [], [], []
            with torch.no_grad():
                for batch in loader:
                    out = model(batch, cfg.loss_weights, tau_geo=cfg.train.temperature_geo)
                    z = out["z"]
                    mu = model.gcatt.prototypes.get()
                    dists = torch.cdist(z, mu, p=2) ** 2
                    probs = torch.softmax(-dists, dim=-1)
                    preds = probs.argmax(dim=-1)
                    all_probs.append(probs.numpy())
                    all_preds.append(preds.numpy())
                    all_true.append(batch["pathology"].numpy())

            y_true = np.concatenate(all_true)
            y_pred = np.concatenate(all_preds)
            y_prob = np.concatenate(all_probs, axis=0)

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            kappa = cohen_kappa_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            try:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            except ValueError:
                auc = float("nan")

            lines = [
                "=" * 60,
                "GACL evaluation -- real metrics, own prototype-distance head",
                "=" * 60,
                f"n = {len(y_true)}",
                f"accuracy   : {acc:.4f}   (chance level: {chance:.4f})",
                f"macro-F1   : {f1:.4f}",
                f"macro-AUC  : {auc:.4f}   (chance level: 0.5000)",
                f"Cohen kappa: {kappa:.4f}   (0 = chance-level agreement)",
                f"MCC        : {mcc:.4f}   (0 = chance-level agreement)",
            ]
            if abs(acc - chance) < 0.03 and abs(kappa) < 0.03:
                lines += ["", "NOTE: within ~3 points of chance level, kappa ~0.",
                          "Report this as GACL's real result on this data, not as a bug."]
            self.text.setPlainText("\n".join(lines))
        except Exception as e:
            self.text.setPlainText(f"Evaluation failed to run: {e}")

    # ------------------------------------------------------------------ #
    def _select_image_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select real image dataset folder (one subfolder per class)")
        if path:
            self.image_root = path
            self.text.append(f"\nReal image folder selected: {path}")

    def _train_gacl_on_real_images(self) -> None:
        """Runs the real GACL training loop (gacl/image_dataset.py + gacl/model.py)
        directly against a real, folder-per-class image dataset, then evaluates on
        the held-out validation and test splits it produces -- and prints whatever
        comes out, next to chance-level context, exactly like train_gacl_real_images.py.
        This call is synchronous and will block the UI for the duration of training;
        for large datasets or many epochs, running train_gacl_real_images.py directly
        from a terminal is faster and does not block the GUI."""
        if not self._torch_available:
            QMessageBox.warning(self, "torch unavailable", "Install torch first (see panel text).")
            return
        if not self.image_root or not os.path.isdir(self.image_root):
            QMessageBox.warning(self, "No image folder", "Select a real image dataset folder first.")
            return
        try:
            import torch
            from torch.utils.data import DataLoader
            from sklearn.metrics import (
                accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef,
                balanced_accuracy_score,
            )
            from gacl.config import make_real_image_config
            from gacl.model import GACLModel
            from gacl.image_dataset import (
                scan_real_image_root, split_records, GACLRealImageDataset,
                CROP_CLASSES_REAL, PATHOLOGY_CLASSES_REAL,
            )
        except ImportError as e:
            self.text.setPlainText(f"GACL real-image training UNAVAILABLE: {e}")
            return

        from PyQt6.QtWidgets import QApplication

        try:
            self.text.setPlainText("Scanning real image dataset...")
            QApplication.processEvents()
            records = scan_real_image_root(self.image_root)
            if not records:
                self.text.append("\nNo images matched the expected folder-per-class layout. "
                                  "See gacl/image_dataset.py's CLASS_TO_CROP_PATHOLOGY table.")
                return
            splits = split_records(records)
            n_classes = len(PATHOLOGY_CLASSES_REAL)
            chance = 1.0 / n_classes

            self.text.append(f"Found {len(records)} unique real images, {n_classes} pathology classes.")
            self.text.append(f"Split -- train: {len(splits['train'])}, validation: {len(splits['validation'])}, "
                              f"test: {len(splits['test'])}")
            self.text.append(f"Chance level ({n_classes} classes): {chance:.4f}")
            self.text.append("Section 6.10's classical-feature benchmark: balanced accuracy 33.8%, macro-F1 35.8%.")
            QApplication.processEvents()

            cfg = make_real_image_config(num_pathology_classes=n_classes)
            epochs = 20
            train_ds = GACLRealImageDataset(splits["train"])
            val_ds = GACLRealImageDataset(splits["validation"])
            test_ds = GACLRealImageDataset(splits["test"])
            train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False)

            device = torch.device("cpu")
            model = GACLModel(cfg, num_crops=len(CROP_CLASSES_REAL), recon_input_dim=cfg.vlae.recon_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

            self.text.append(f"\nTraining for {epochs} epochs (real forward/backward pass)...")
            QApplication.processEvents()
            for epoch in range(epochs):
                model.train()
                running, n_batches = {}, 0
                for batch in train_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    optimizer.zero_grad()
                    out = model(batch, cfg.loss_weights, tau_geo=cfg.train.temperature_geo)
                    out["loss_total"].backward()
                    optimizer.step()
                    n_batches += 1
                    for k, v in out.items():
                        if k.startswith("loss_"):
                            running[k] = running.get(k, 0.0) + v.item()
                avg = {k: v / n_batches for k, v in running.items()}
                self.text.append(f"Epoch {epoch+1}/{epochs}  " + "  ".join(f"{k}={v:.4f}" for k, v in avg.items()))
                QApplication.processEvents()

            def _evaluate(loader, split_name):
                model.eval()
                all_probs, all_preds, all_true = [], [], []
                with torch.no_grad():
                    for batch in loader:
                        out = model(batch, cfg.loss_weights, tau_geo=cfg.train.temperature_geo)
                        z = out["z"]
                        mu = model.gcatt.prototypes.get()
                        dists = torch.cdist(z, mu, p=2) ** 2
                        probs = torch.softmax(-dists, dim=-1)
                        all_probs.append(probs.numpy())
                        all_preds.append(probs.argmax(dim=-1).numpy())
                        all_true.append(batch["pathology"].numpy())
                import numpy as np
                y_true = np.concatenate(all_true)
                y_pred = np.concatenate(all_preds)
                acc = accuracy_score(y_true, y_pred)
                bal_acc = balanced_accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
                kappa = cohen_kappa_score(y_true, y_pred)
                mcc = matthews_corrcoef(y_true, y_pred)
                self.text.append(f"\n[{split_name}] n={len(y_true)}  accuracy={acc:.4f} (chance={chance:.4f})  "
                                  f"balanced_acc={bal_acc:.4f}  macro-F1={f1:.4f}  kappa={kappa:.4f}  mcc={mcc:.4f}")
                if abs(bal_acc - chance) < 0.05 and abs(kappa) < 0.05:
                    self.text.append(f"  NOTE: within ~5 points of chance on {split_name}. This is GACL's honest "
                                      f"result here, not a bug -- compare against the 33.8% classical benchmark.")
                else:
                    self.text.append(f"  Meaningfully above chance -- compare directly against the 33.8% "
                                      f"classical-feature benchmark (Section 6.10) to see which is stronger.")

            _evaluate(val_loader, "validation")
            _evaluate(test_loader, "test")
            self.text.append("\nDone.")
        except Exception as e:
            self.text.append(f"\nTraining/evaluation failed: {e}")
