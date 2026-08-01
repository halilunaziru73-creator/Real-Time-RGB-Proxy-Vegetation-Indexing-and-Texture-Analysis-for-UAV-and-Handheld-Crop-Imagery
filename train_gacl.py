"""
train.py

Runs the GACL composite objective (Section 7.8) over GACL_Data.xlsx.

READ THIS FIRST
----------------
This script will happily run and print a decreasing loss curve. That is
expected of *any* sufficiently flexible network on *any* data -- on its own
it is not evidence that GACL has learned a real pathology-relevant
representation. This script does NOT print or save any accuracy/F1/AUC-style
"performance" claim, because that would require confirming the label quality
and collection protocol behind GACL_Data.xlsx, which is outside what this
code can verify on its own. It prints raw loss component values only, as a
code-correctness / smoke-test signal, clearly labeled as such.

If you've confirmed the labels are trustworthy ground truth from a real
measurement/collection protocol, see README.md for how to extend
evaluate.py with real accuracy/F1 reporting.

Usage:
    python train.py --data /path/to/GACL_Data.xlsx --epochs 5
"""

import argparse
import torch
from torch.utils.data import DataLoader

from gacl import GACLConfig, GACLModel, GACLTabularDataset, CROP_CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to GACL_Data.xlsx")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = GACLConfig()
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    cfg.train.device = args.device

    print("=" * 78)
    print("GACL reference training run -- SMOKE TEST ONLY, NOT A BENCHMARK.")
    print("Label quality / collection protocol for GACL_Data.xlsx has not")
    print("been independently verified here. Loss values below confirm the")
    print("code runs and the math is wired correctly -- nothing more.")
    print("=" * 78)

    train_ds = GACLTabularDataset(args.data, split="train",
                                   num_patches=cfg.hgavit.num_patches,
                                   patch_dim=cfg.hgavit.patch_dim)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                               drop_last=True)

    device = torch.device(args.device)
    model = GACLModel(cfg, num_crops=len(CROP_CLASSES),
                       recon_input_dim=cfg.vlae.recon_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                               weight_decay=cfg.train.weight_decay)

    model.train()
    for epoch in range(cfg.train.epochs):
        running = {}
        n_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch, cfg.loss_weights, tau_geo=cfg.train.temperature_geo)

            optim.zero_grad()
            out["loss_total"].backward()
            optim.step()

            n_batches += 1
            for k, v in out.items():
                if k.startswith("loss_"):
                    running[k] = running.get(k, 0.0) + v.item()

        avg = {k: v / n_batches for k, v in running.items()}
        print(f"[epoch {epoch+1}/{cfg.train.epochs}] " +
              " ".join(f"{k}={v:.4f}" for k, v in avg.items()))

    torch.save(model.state_dict(), "gacl_smoketest_checkpoint.pt")
    print("\nSaved checkpoint to gacl_smoketest_checkpoint.pt")
    print("Reminder: this checkpoint has not been validated against any")
    print("real-world capability claim. See README.md and paper Section 10")
    print("for the data-collection protocol required before that is possible.")


if __name__ == "__main__":
    main()
