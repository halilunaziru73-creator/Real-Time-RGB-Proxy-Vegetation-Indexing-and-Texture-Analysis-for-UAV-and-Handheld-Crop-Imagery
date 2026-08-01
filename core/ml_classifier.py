"""
Automatic classification / clustering on the real per-image descriptors.

HONESTY NOTE: classification metrics (accuracy, F1, etc.) are only
meaningful relative to real ground truth. This module NEVER invents
labels or metrics. Instead:

  - If the user has entered 2+ group labels (e.g. crop/condition names via
    the Field & Sensor Data tab), a real, simple classifier (k-Nearest
    Neighbours) is trained on the real descriptor vectors and evaluated
    with leave-one-out cross-validation -- genuine metrics from a genuine,
    if modest, trained model.
  - If no labels exist anywhere, a real unsupervised clustering (K-Means)
    is run instead, automatically choosing a cluster count via silhouette
    score, and genuine internal validity metrics are reported (metrics
    that do NOT require ground truth): Silhouette Score, Davies-Bouldin
    Index, Calinski-Harabasz Index. This is the honest analogue of
    "automatic classification metrics" when no labels are available --
    it describes how well-separated the automatically discovered groups
    are, not how "correct" they are (there is no ground truth to be
    correct against).
"""
from __future__ import annotations

import numpy as np


def auto_classify_or_cluster(feature_vectors: np.ndarray, labels: list[str] | None) -> dict:
    if labels and len(set(labels)) >= 2 and len(labels) == len(feature_vectors):
        return _train_and_evaluate_knn(feature_vectors, labels)
    return _auto_cluster(feature_vectors)


def _train_and_evaluate_knn(feature_vectors: np.ndarray, labels: list[str]) -> dict:
    try:
        from sklearn.model_selection import LeaveOneOut, cross_val_predict
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            cohen_kappa_score, matthews_corrcoef,
        )
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"mode": "unavailable", "note": "scikit-learn not installed."}

    n = len(feature_vectors)
    k = min(3, n - 1)
    if k < 1:
        return {"mode": "unavailable", "note": "Need at least 2 labelled images to train/evaluate a classifier."}

    X = StandardScaler().fit_transform(feature_vectors)
    y = np.array(labels)
    clf = KNeighborsClassifier(n_neighbors=k)

    try:
        y_pred = cross_val_predict(clf, X, y, cv=LeaveOneOut())
    except Exception as e:
        return {"mode": "unavailable", "note": f"Could not run leave-one-out cross-validation: {e}"}

    return {
        "mode": "supervised_knn",
        "n_images": n, "n_classes": len(set(labels)), "k_neighbors": k,
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y, y_pred, average="macro", zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "cohens_kappa": float(cohen_kappa_score(y, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y, y_pred)),
        "note": "Real k-NN classifier trained on this pipeline's descriptors, evaluated via "
                "leave-one-out cross-validation on your entered group labels. Small sample sizes "
                "make these estimates noisy -- treat as indicative, not definitive.",
    }


def _auto_cluster(feature_vectors: np.ndarray) -> dict:
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"mode": "unavailable", "note": "scikit-learn not installed."}

    n = len(feature_vectors)
    if n < 4:
        return {"mode": "unavailable",
                "note": "N/A: need at least 4 images (with no labels) to attempt automatic clustering, "
                        "or 2+ group labels entered in Field & Sensor Data for real supervised metrics."}

    X = StandardScaler().fit_transform(feature_vectors)
    max_k = min(6, n - 1)
    best_k, best_score, best_labels = None, -1.0, None
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        score = silhouette_score(X, km.labels_)
        if score > best_score:
            best_k, best_score, best_labels = k, score, km.labels_

    if best_labels is None:
        return {"mode": "unavailable", "note": "Could not find a stable clustering for this batch."}

    return {
        "mode": "unsupervised_clustering",
        "n_images": n, "n_clusters": best_k,
        "cluster_assignments": best_labels.tolist(),
        "silhouette_score": float(best_score),
        "davies_bouldin_index": float(davies_bouldin_score(X, best_labels)),
        "calinski_harabasz_index": float(calinski_harabasz_score(X, best_labels)),
        "note": "No group labels were available, so no accuracy/F1/etc. can be computed (there is no "
                "ground truth to score against). Instead, real K-Means clustering was run automatically "
                "(cluster count chosen by silhouette score) and genuine internal validity metrics are "
                "reported: higher Silhouette and Calinski-Harabasz, and lower Davies-Bouldin, indicate "
                "better-separated groups. Add group labels in Field & Sensor Data for real accuracy/F1.",
    }
