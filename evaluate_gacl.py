"""
evaluate.py

Loads a checkpoint produced by train.py and runs it over the 'test' split of
GACL_Data.xlsx.

ON PRINTING ACCURACY / F1 / AUC
-------------------------------
Earlier versions of this script deliberately withheld these numbers, because
check_learnable_structure.py had already shown the accompanying data carries
no confirmed learnable feature-label structure -- a "high" number would most
likely reflect the classifier head fitting incidental correlations or noise,
not genuine pathology-recognition capability.

This version computes and prints the real numbers anyway, using the
prototype-distance classification head that GCATT already maintains
(Section 7.5): predicted class = argmax over negative squared distance to
each pathology prototype, exactly the same quantity `GCATT.prototype_loss`
trains against. Nothing here is a proxy computed on some other model --
these are GACL's own predictions, scored with standard sklearn metrics.

We report them WITH their chance-level context printed alongside, every
time, rather than as a bare number: a bare "84%" and a bare "20%" look
identical in a table unless the reader is told what the class-balanced
chance level is. If you see a number near chance level here, that is GACL's
honest result on this dataset, not a bug in this script and not a reason to
omit the number -- report it exactly as printed.

Usage:
    python evaluate.py --data /path/to/GACL_Data.xlsx --checkpoint gacl_smoketest_checkpoint.pt
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, cohen_kappa_score, matthews_corrcoef,
)

from gacl import GACLConfig, GACLModel, GACLTabularDataset, CROP_CLASSES, PATHOLOGY_CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = GACLConfig()
    device = torch.device(args.device)

    test_ds = GACLTabularDataset(args.data, split="test",
                                  num_patches=cfg.hgavit.num_patches,
                                  patch_dim=cfg.hgavit.patch_dim)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    n_classes = len(PATHOLOGY_CLASSES)
    chance_level = 1.0 / n_classes

    model = GACLModel(cfg, num_crops=len(CROP_CLASSES),
                       recon_input_dim=cfg.vlae.recon_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    print("=" * 78)
    print("GACL evaluation run: real prototype-distance classification metrics,")
    print(f"reported alongside chance level ({chance_level:.4f} for {n_classes} classes).")
    print("A result at or near chance level is GACL's honest result on this")
    print("dataset -- report it as printed, not as a validated capability.")
    print("=" * 78)

    running = {}
    n_batches = 0
    all_probs, all_preds, all_true = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch, cfg.loss_weights, tau_geo=cfg.train.temperature_geo)
            n_batches += 1
            for k, v in out.items():
                if k.startswith("loss_"):
                    running[k] = running.get(k, 0.0) + v.item()

            # Real classification head: negative squared distance to each
            # pathology prototype (same logits GCATT.prototype_loss trains
            # against), softmax'd to class probabilities.
            z = out["z"]
            mu = model.gcatt.prototypes.get()
            dists = torch.cdist(z, mu, p=2) ** 2
            logits = -dists
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_true.append(batch["pathology"].cpu().numpy())

    avg = {k: v / n_batches for k, v in running.items()}
    print("\n[test split, diagnostic loss values]")
    for k, v in avg.items():
        print(f"  {k}: {v:.4f}")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs, axis=0)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError as e:
        auc = float("nan")
        print(f"  (AUC unavailable: {e})")

    print(f"\n[test split, real classification metrics -- n={len(y_true)}]")
    print(f"  accuracy         : {acc:.4f}   (chance level: {chance_level:.4f})")
    print(f"  macro-precision  : {prec:.4f}")
    print(f"  macro-recall     : {rec:.4f}")
    print(f"  macro-F1         : {f1:.4f}")
    print(f"  macro-AUC (OvR)  : {auc:.4f}   (chance level: 0.5000)")
    print(f"  Cohen's kappa    : {kappa:.4f}   (0 = chance-level agreement)")
    print(f"  Matthews corrcoef: {mcc:.4f}   (0 = chance-level agreement)")
    print("\n[confusion matrix, rows=true, cols=predicted, class order:", PATHOLOGY_CLASSES, "]")
    print(confusion_matrix(y_true, y_pred))

    if abs(acc - chance_level) < 0.03 and abs(kappa) < 0.03:
        print("\nNOTE: accuracy sits within ~3 points of chance level and kappa is ~0.")
        print("This matches check_learnable_structure.py's independent finding for")
        print("this dataset. Report this result as-is; it is not evidence GACL")
        print("recognises real pathology signal on this data.")


if __name__ == "__main__":
    main()

