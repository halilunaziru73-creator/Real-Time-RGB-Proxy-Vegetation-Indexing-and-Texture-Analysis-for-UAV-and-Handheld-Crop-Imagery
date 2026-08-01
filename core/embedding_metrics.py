"""
Machine-learning / representation metrics.

HONESTY NOTE: this pipeline does not train a contrastive/SSL encoder.
"Embedding" here means the real, simple 8-dimensional colour/texture/NDVI
descriptor already extracted per image (see core.indices.feature_vector).
Metrics below are computed genuinely on those descriptors and on real
image transforms -- they are legitimate, just modest in scope compared to
a trained deep embedding model. Metrics that require labelled ground
truth (accuracy, precision/recall/F1, ROC-AUC, mAP, Cohen's Kappa, MCC,
transfer/zero-shot/few-shot accuracy) are reported only if the user
supplies real labels/predictions; otherwise they are marked N/A rather
than invented.
"""
from __future__ import annotations

import numpy as np

from .indices import feature_vector


def embedding_dimensionality() -> int:
    """Real dimensionality of this pipeline's descriptor vector (not a deep embedding)."""
    return 8


def pairwise_metrics(descriptors: np.ndarray) -> dict:
    """Cosine similarity and Euclidean distance matrices across real descriptors."""
    feats = np.asarray(descriptors)
    norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    cosine_sim = norm @ norm.T

    diff = feats[:, None, :] - feats[None, :, :]
    euclidean = np.sqrt((diff ** 2).sum(axis=2))

    return {"cosine_similarity": cosine_sim, "euclidean_distance": euclidean, "feature_vectors": feats}


def latent_feature_variance(descriptors: np.ndarray) -> float:
    """Variance of the real descriptor vectors across the batch (a genuine, if simple, latent-space spread metric)."""
    feats = np.asarray(descriptors)
    return float(np.var(feats, axis=0).mean())


def intra_inter_similarity(descriptors: np.ndarray, labels: list[str] | None) -> dict:
    """
    Intra-class vs inter-class cosine similarity. Requires group labels
    (e.g. crop type) supplied by the user via the Field & Sensor Data
    panel; without labels this is not computable and is reported as such.
    """
    if not labels or len(set(labels)) < 2:
        return {"available": False,
                "note": "N/A: needs 2+ group labels (e.g. crop type) entered in Field & Sensor Data."}
    feats = np.asarray(descriptors)
    norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    sim = norm @ norm.T
    labels = np.array(labels)
    intra, inter = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            (intra if labels[i] == labels[j] else inter).append(sim[i, j])
    return {
        "available": True,
        "intra_class_similarity": float(np.mean(intra)) if intra else float("nan"),
        "inter_class_similarity": float(np.mean(inter)) if inter else float("nan"),
    }


def info_nce_self_supervised(image_array: np.ndarray, temperature: float = 0.5) -> float:
    """
    A genuine, self-contained InfoNCE-style contrastive loss computed on this
    pipeline's real descriptor: the positive pair is (image, horizontally-
    flipped image); negatives are the descriptor shifted by random per-
    dimension jitter (a stand-in "batch" since no real batch of distinct
    instances/augmentations is available here). This demonstrates the loss
    mechanics on real data; it is NOT the loss of a trained SSL model.
    """
    flipped = image_array[:, ::-1, :]
    anchor = feature_vector(image_array)
    positive = feature_vector(flipped)

    rng = np.random.default_rng(int(anchor.sum()) % (2 ** 32))
    negatives = anchor[None, :] + rng.normal(scale=anchor.std() + 1e-6, size=(8, anchor.shape[0]))

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    pos_sim = cos(anchor, positive) / temperature
    neg_sims = np.array([cos(anchor, neg) / temperature for neg in negatives])
    logits = np.concatenate([[pos_sim], neg_sims])
    loss = -pos_sim + np.log(np.sum(np.exp(logits)))
    return float(loss)


def invariance_tests(image_array: np.ndarray) -> dict:
    """
    Genuine self-supervised invariance check: apply real geometric
    transforms to the image, recompute the real descriptor, and measure
    cosine similarity to the original descriptor. 1.0 = perfectly
    invariant under that transform; lower = descriptor changed.
    """
    anchor = feature_vector(image_array)

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    rot90 = np.rot90(image_array, k=1)
    rot180 = np.rot90(image_array, k=2)
    flip_h = image_array[:, ::-1, :]
    flip_v = image_array[::-1, :, :]
    h, w = image_array.shape[:2]
    scaled = image_array[h // 4: 3 * h // 4, w // 4: 3 * w // 4, :]  # centre-crop as a scale/viewpoint proxy
    shifted = np.roll(image_array, shift=(h // 10, w // 10), axis=(0, 1))  # translation

    return {
        "rotation_90": cos(anchor, feature_vector(rot90)),
        "rotation_180": cos(anchor, feature_vector(rot180)),
        "horizontal_flip": cos(anchor, feature_vector(flip_h)),
        "vertical_flip": cos(anchor, feature_vector(flip_v)),
        "scale_viewpoint_crop": cos(anchor, feature_vector(scaled)),
        "translation": cos(anchor, feature_vector(shifted)),
    }


def cross_image_similarity_recommendation(loaded_names: list[str], feature_vectors: np.ndarray) -> dict:
    """
    Cross-image relational analysis: real cosine similarity between this
    pipeline's descriptors, used to suggest which images' findings might
    transfer to which others (e.g. "if image 3 was confirmed diseased,
    image 7 looks most similar and may be worth inspecting next").

    HONESTY NOTE: this is a real similarity heuristic on simple descriptors,
    not a trained cross-crop pathotype-alignment model. No labelled
    multi-crop pathology dataset exists here to train such a model, so no
    "Pathotype Alignment" or "Cross-Crop Manifold Distance" score (which
    would require one) is fabricated.
    """
    n = len(loaded_names)
    if n < 2:
        return {"available": False, "note": "Needs at least 2 images to compare."}

    norm = feature_vectors / (np.linalg.norm(feature_vectors, axis=1, keepdims=True) + 1e-9)
    sim = norm @ norm.T
    np.fill_diagonal(sim, -np.inf)

    recommendations = []
    for i in range(n):
        j = int(np.argmax(sim[i]))
        recommendations.append({
            "image": loaded_names[i],
            "most_similar_image": loaded_names[j],
            "similarity": float(sim[i, j]),
        })
    return {
        "available": True,
        "recommendations": recommendations,
        "note": "Real cosine similarity between images' colour/texture/NDVI descriptors. High similarity "
                "suggests two images share a comparable visual/vigor profile and inspection findings on one "
                "may be worth checking on the other. This is a descriptor-similarity heuristic, not a trained "
                "cross-crop pathotype-alignment model.",
    }


def classification_metrics_from_labels(y_true: list, y_pred: list) -> dict:
    """
    Real classification metrics, computed ONLY if the user supplies actual
    ground-truth and predicted labels (e.g. via a CSV upload). Never
    fabricated.
    """
    try:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            cohen_kappa_score, matthews_corrcoef,
        )
    except ImportError:
        return {"available": False, "note": "scikit-learn not installed."}

    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {"available": False, "note": "N/A: requires matched true/predicted label lists."}

    average = "macro"  # safe for binary or multiclass, regardless of label type (works with string labels too)
    return {
        "available": True,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "cohens_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "note": "ROC-AUC and mAP need predicted probabilities/rankings, not just hard labels; "
                "supply those separately if needed.",
    }
