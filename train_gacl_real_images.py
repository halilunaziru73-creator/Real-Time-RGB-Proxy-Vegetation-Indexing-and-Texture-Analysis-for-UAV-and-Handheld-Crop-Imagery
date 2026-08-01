"""
train_gacl_real_images.py

Trains and evaluates GACL on real crop-disease photographs (gacl/image_dataset.py),
replacing the tabular pseudo-patch stand-in used elsewhere in this repo.

WHAT THIS SCRIPT DOES AND DOES NOT ESTABLISH
------------------------------------------------
This is the first GACL training/evaluation run in this project against real
image data. Section 6.9-6.12 of the accompanying paper already showed that
classical hand-crafted features (GLCM + colour/NDVI-proxy) find real,
above-chance classifiable structure in this same dataset (RandomForest
balanced accuracy 33.8% vs. 4.55% chance, robustness-checked against
augmentation leakage). This script tests whether GACL's own learned
representation can do the same or better -- a genuinely open question this
script answers empirically rather than assumes.

Print behaviour matches evaluate_gacl.py's honesty rule exactly: real
accuracy/macro-F1/macro-AUC/kappa/MCC are computed and printed, always next
to their chance-level reference, with an explicit flag if the result lands
within noise of chance. No number is suppressed or dressed up either way.

USAGE
-----
    python train_gacl_real_images.py --data_root /path/to/My_Data --epochs 20

We were unable to execute this script ourselves in the environment used to
prepare the accompanying paper (no network access there to install PyTorch).
It is included so it can be run in an environment that has torch installed,
and the honest output -- whatever it is -- can be reported back.
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix,
    balanced_accuracy_score,
)

from gacl.config import make_real_image_config
from gacl.model import GACLModel
from gacl.image_dataset import (
    scan_real_image_root, split_records, GACLRealImageDataset,
    CROP_CLASSES_REAL, PATHOLOGY_CLASSES_REAL,
)


def evaluate(model, loader, device, chance_level, split_name):
    model.eval()
    all_probs, all_preds, all_true = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch, model.cfg.loss_weights if hasattr(model, "cfg") else None,
                        tau_geo=0.07)
            z = out["z"]
            mu = model.gcatt.prototypes.get()
            dists = torch.cdist(z, mu, p=2) ** 2
            probs = torch.softmax(-dists, dim=-1)
            preds = probs.argmax(dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_true.append(batch["pathology"].cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs, axis=0)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError as e:
        auc = float("nan")

    print(f"\n[{split_name} split -- real GACL metrics, n={len(y_true)}]")
    print(f"  accuracy          : {acc:.4f}   (chance level: {chance_level:.4f})")
    print(f"  balanced accuracy : {bal_acc:.4f}   (chance level: {chance_level:.4f})")
    print(f"  macro-F1          : {f1:.4f}")
    print(f"  macro-AUC (OvR)   : {auc:.4f}   (chance level: 0.5000)")
    print(f"  Cohen's kappa     : {kappa:.4f}   (0 = chance-level agreement)")
    print(f"  Matthews corrcoef : {mcc:.4f}   (0 = chance-level agreement)")

    if abs(bal_acc - chance_level) < 0.05 and abs(kappa) < 0.05:
        print(f"  NOTE: within ~5 points of chance level on {split_name}; report as GACL's honest")
        print(f"        result on this data, not as a bug. Section 6.10's classical-feature result")
        print(f"        (33.8% balanced accuracy) is the benchmark to compare this against.")
    else:
        print(f"  This is meaningfully above chance level. Compare directly against Section 6.10's")
        print(f"  classical-feature Random Forest result (33.8% balanced accuracy) to see whether")
        print(f"  GACL's learned representation outperforms the hand-crafted features.")

    return dict(accuracy=acc, balanced_accuracy=bal_acc, macro_f1=f1, macro_auc=auc,
                kappa=kappa, mcc=mcc, y_true=y_true, y_pred=y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                         help="Path to the folder-per-class real image dataset (e.g. My_Data/)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_checkpoint", type=str, default="gacl_realimage_checkpoint.pt")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Scanning real image dataset...")
    records = scan_real_image_root(args.data_root)
    print(f"Found {len(records)} unique real images across "
          f"{len(set(r.pathology for r in records))} pathology classes, "
          f"{len(set(r.crop for r in records))} crops.")

    splits = split_records(records)
    print(f"Split sizes -- train: {len(splits['train'])}, validation: {len(splits['validation'])}, "
          f"test: {len(splits['test'])}")

    cfg = make_real_image_config(num_pathology_classes=len(PATHOLOGY_CLASSES_REAL),
                                  image_size=args.image_size, patch_size=args.patch_size)
    cfg.train.batch_size = args.batch_size
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.train.device = args.device

    train_ds = GACLRealImageDataset(splits["train"], image_size=args.image_size, patch_size=args.patch_size)
    val_ds = GACLRealImageDataset(splits["validation"], image_size=args.image_size, patch_size=args.patch_size)
    test_ds = GACLRealImageDataset(splits["test"], image_size=args.image_size, patch_size=args.patch_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = GACLModel(cfg, num_crops=len(CROP_CLASSES_REAL), recon_input_dim=cfg.vlae.recon_dim).to(device)
    model.cfg = cfg  # convenience reference used by evaluate() above
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=cfg.train.weight_decay)

    chance_level = 1.0 / len(PATHOLOGY_CLASSES_REAL)
    print(f"\nChance level for {len(PATHOLOGY_CLASSES_REAL)} pathology classes: {chance_level:.4f}")
    print("Section 6.10's classical-feature benchmark (Random Forest, held-out test): "
          "balanced accuracy 33.8%, macro-F1 35.8%. This run's job is to honestly report "
          "whether GACL's learned representation matches, exceeds, or falls short of that.\n")

    print("=" * 78)
    print("Training GACL on real images (real forward/backward pass -- loss values only)")
    print("=" * 78)
    for epoch in range(cfg.train.epochs):
        model.train()
        running = {}
        n_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(batch, cfg.loss_weights, tau_geo=cfg.train.temperature_geo)
            loss = out["loss_total"]
            loss.backward()
            optimizer.step()
            n_batches += 1
            for k, v in out.items():
                if k.startswith("loss_"):
                    running[k] = running.get(k, 0.0) + v.item()
        avg = {k: v / n_batches for k, v in running.items()}
        print(f"Epoch {epoch+1}/{cfg.train.epochs}  " + "  ".join(f"{k}={v:.4f}" for k, v in avg.items()))

    torch.save(model.state_dict(), args.save_checkpoint)
    print(f"\nCheckpoint saved to {args.save_checkpoint}")

    evaluate(model, val_loader, device, chance_level, "validation")
    evaluate(model, test_loader, device, chance_level, "test")


if __name__ == "__main__":
    main()
