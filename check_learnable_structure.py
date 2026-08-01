"""
check_learnable_structure.py

An empirical (not textual) check of whether GACL_Data.xlsx contains
feature-label structure a real classifier can exploit. This does not rely on
any claim from the paper or from anyone's description of the data -- it just
trains strong, standard classifiers (RandomForest) against every numeric
feature set in the file and compares test-set performance to a
stratified-random baseline and to chance level.

Usage:
    python check_learnable_structure.py --data GACL_Data.xlsx

Output: a table of test accuracy / macro-F1 per feature set, plus baselines,
printed to stdout. Nothing here is hard-coded -- rerun it any time the
dataset changes and the numbers will change with it.
"""

import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

EMBED_COLS = [f"embedding_{i}" for i in range(1, 33)]
LATENT_COLS = [f"latent_z{i}" for i in range(1, 17)]
GEO_COLS = ["camera_height_m", "camera_pitch_deg", "camera_roll_deg",
            "camera_yaw_deg", "distance_m"]
RAW_VIS_COLS = ["mean_R", "mean_G", "mean_B", "ExG", "VARI",
                "GLCM_contrast", "GLCM_entropy", "area", "perimeter", "solidity"]

FEATURE_SETS = {
    "embeddings_only": EMBED_COLS,
    "raw_visual_features_only": RAW_VIS_COLS,
    "embeddings+latents+geo": EMBED_COLS + LATENT_COLS + GEO_COLS,
    "all_numeric_features": EMBED_COLS + LATENT_COLS + GEO_COLS + RAW_VIS_COLS,
}


def run_check(data_path, target_col="pathology", n_estimators=300, random_state=42):
    df = pd.read_excel(data_path)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    y_train = train_df[target_col]
    y_test = test_df[target_col]

    print(f"Target column: {target_col}")
    print(f"Train rows: {len(train_df)}   Test rows: {len(test_df)}")
    print("Test-set class distribution:")
    print(y_test.value_counts().to_string())
    n_classes = y_test.nunique()
    print(f"Chance level for {n_classes} balanced classes: {1/n_classes:.4f}\n")

    results = {}
    for name, cols in FEATURE_SETS.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        scaler = StandardScaler().fit(train_df[cols].values)
        X_train = scaler.transform(train_df[cols].values)
        X_test = scaler.transform(test_df[cols].values)

        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        results[name] = (acc, f1)
        print(f"[RandomForest | {name}] test accuracy={acc:.4f}  macro-F1={f1:.4f}")

    dummy_cols = FEATURE_SETS["embeddings_only"]
    dummy_cols = [c for c in dummy_cols if c in df.columns]
    dummy = DummyClassifier(strategy="stratified", random_state=random_state)
    dummy.fit(train_df[dummy_cols], y_train)
    pred_dummy = dummy.predict(test_df[dummy_cols])
    acc_dummy = accuracy_score(y_test, pred_dummy)
    f1_dummy = f1_score(y_test, pred_dummy, average="macro")
    print(f"\n[DummyClassifier stratified-random baseline] "
          f"test accuracy={acc_dummy:.4f}  macro-F1={f1_dummy:.4f}")

    return results, (acc_dummy, f1_dummy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, required=True, help="Path to GACL_Data.xlsx")
    parser.add_argument("--target", type=str, default="pathology",
                         choices=["pathology", "crop"],
                         help="Label column to test for learnable structure against")
    args = parser.parse_args()

    print("=" * 78)
    print("Empirical learnable-structure check -- results depend only on the")
    print("data file passed in, not on any claim in accompanying documentation.")
    print("=" * 78 + "\n")

    run_check(args.data, target_col=args.target)

    print("\nHow to read this: if every feature set's accuracy/macro-F1 sits")
    print("within noise of the stratified-random baseline and of 1/num_classes")
    print("chance level, a standard strong classifier (RandomForest) found no")
    print("usable signal in these columns for this target. That's an empirical")
    print("result specific to this file -- rerun against any updated or")
    print("different dataset to get a fresh answer.")


if __name__ == "__main__":
    main()
