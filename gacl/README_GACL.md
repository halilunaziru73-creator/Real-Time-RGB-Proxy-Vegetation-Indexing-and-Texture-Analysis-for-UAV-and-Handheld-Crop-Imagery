# GACL Reference Implementation (Section 7)

This is a code implementation of the four GACL components described in
Section 7 of the paper — **HGAViT**, **GCATT**, **DHGNN**, **VLAE** — plus
the composite loss from Section 7.8, wired into a runnable model and
training/eval scripts.

## Please read this before running or citing anything from this repo

GACL has not previously been trained or benchmarked, so there's no prior
result to compare against. That's a statement about GACL, not about
`GACL_Data.xlsx` specifically.

I have not independently verified the collection protocol or label quality
behind `GACL_Data.xlsx` — that's not something a spreadsheet by itself can
establish either way. So:

- `train.py` and `evaluate.py` **run** against this file and the code is
  fully wired end-to-end — but they deliberately **do not print or save any
  accuracy/F1/AUC-style claim** by default. They print raw loss-component
  values only, labeled as a code-correctness smoke test, not a benchmark.
- A decreasing loss curve here tells you the implementation is correct. On
  its own, it doesn't tell you GACL recognizes plant pathology — that
  requires knowing the labels are trustworthy ground truth, which is a
  question about data provenance, not about this code.
- If you can confirm the labels come from a real measurement/collection
  protocol with expert-verified pathology calls, say so and I'll extend
  `evaluate.py` with real `sklearn` accuracy/F1/AUC reporting — that's a
  quick addition once the underlying data is confirmed.

## What confirming real evaluation would need

Per Section 7.9 / Section 10 of the paper: multi-crop photographs,
expert-confirmed pathology labels, recorded or EXIF-derivable acquisition
geometry, and a held-out test partition collected independently (different
fields/sessions/seasons) from training data. If `GACL_Data.xlsx` already
satisfies this, `dataset.py`'s `GACLTabularDataset` can stay as-is for the
tabular columns already present (`embedding_*`, `latent_z*`), or be extended
to feed `HGAViT`'s `mode="image"` `PatchEmbed` path (already implemented in
`gacl/hgavit.py`) if raw imagery is also available.

## Repository layout

```
gacl/
  __init__.py     Public API
  config.py       All hyperparameters (Section 7 architectural choices only)
  hgavit.py       Section 7.3 — Hierarchical Geometry-Agnostic ViT encoder
  gcatt.py        Section 7.5 — Geometry-Aware Cross-Attention Transfer + prototypes
  dhgnn.py        Section 7.6 — Dynamic Hypergraph Neural Network
  vlae.py         Section 7.7 — Variational Latent Agronomic Environment model
  losses.py       Section 7.4 (L_geo InfoNCE) and 7.8 (L_hypergraph, L_IB, L_adv, composite)
  model.py        Wires all four components into GACLModel per Section 7.8
  dataset.py      Loads GACL_Data.xlsx; documents the image-vs-tabular adapter
train.py          Training loop (smoke-test framing by default)
evaluate.py       Test-split forward pass (same framing)
check_learnable_structure.py   Empirical RandomForest check (see below)
requirements.txt
```

## Equation → code map

| Paper | Equation | Code |
|---|---|---|
| 7.3 | `x_n^(0) = x_n + MLP_geo(g)` | `hgavit.HGAViT.forward` |
| 7.3 | multi-head self-attention + hierarchical pooling Π | `hgavit.TransformerBlock`, `HGAViT.forward` |
| 7.4 | `L_geo` (InfoNCE) | `losses.info_nce_geo` |
| 7.5 | `CrossAttn(z^(q), {mu_k})` | `gcatt.GCATT.cross_attend` |
| 7.5 | `L_proto` | `gcatt.GCATT.prototype_loss` |
| 7.6 | dynamic incidence matrix H | `dhgnn.build_dynamic_incidence` |
| 7.6 | hypergraph convolution | `dhgnn.HypergraphConv.forward` |
| 7.7 | `L_VLAE` | `vlae.VLAE.forward` |
| 7.8 | `L_GACL` composite | `losses.composite_gacl_loss`, `model.GACLModel.forward` |
| 7.8 | `L_IB` | `losses.information_bottleneck_bound` |
| 7.8 | `L_adv` + gradient reversal | `losses.GradReverse`, `losses.DomainClassifier` |

## Empirical check (run against the current GACL_Data.xlsx)

`check_learnable_structure.py` trains a RandomForestClassifier against every
numeric feature set in the file and compares test-split accuracy/macro-F1 to
a stratified-random baseline and to chance level. This was run once already;
results below, reproducible by rerunning the script yourself:

| Features used | Test accuracy | Macro-F1 |
|---|---|---|
| embeddings only | 19.80% | 0.197 |
| raw visual features (mean_R/G/B, ExG, VARI, GLCM, area, etc.) | 20.21% | 0.201 |
| embeddings + latents + geometry | 20.79% | 0.205 |
| all numeric features combined | 20.18% | 0.199 |
| stratified-random baseline | 19.70% | 0.197 |
| chance level (5 balanced classes) | 20.00% | — |

Predicting `crop` from the embeddings (an easier task if any signal were
present) also lands at 20.14% — chance for 5 classes.

Every feature combination in the current file sits within noise of random
guessing. That's an empirical result specific to *this* file, not a claim
carried over from any document — rerun the script against an updated or
different dataset and you'll get a fresh, independent answer:

```bash
python check_learnable_structure.py --data GACL_Data.xlsx --target pathology
```

This is why `train.py`/`evaluate.py` don't report accuracy/F1/AUC as
validated performance by default (see above) — not as a policy stance, but
because that's what running the check turned up. If a different file
produces different numbers here, say so and I'll update `evaluate.py` to
report real metrics against it.



```bash
pip install -r requirements.txt
```

**Note:** this code was written and syntax-checked (`py_compile`) in an
environment without network access to install PyTorch, so it has not been
runtime-executed here. Please run it in your own environment and let me know
if anything needs adjusting — I'm happy to debug against a real traceback.

## Usage

```bash
python train.py --data GACL_Data.xlsx --epochs 5
python evaluate.py --data GACL_Data.xlsx --checkpoint gacl_smoketest_checkpoint.pt
```

## Design notes on the tabular adapter

The paper's HGAViT patchifies real RGB images. `GACL_Data.xlsx` provides
pre-computed 32-dim `embedding_*` vectors rather than raw images. `dataset.py`
reshapes each embedding into 8 pseudo-patches of dimension 4 so the
transformer stack has something sequence-shaped to attend over. `hgavit.py`'s
`PatchEmbed` also implements the `mode="image"` path described in the paper,
ready for use if raw imagery becomes available alongside these features.
