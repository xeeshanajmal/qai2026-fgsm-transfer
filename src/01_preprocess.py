"""Build the train, validation and test splits from CICIDS2017.

Reads the Friday afternoon DDoS capture, reduces it to four features, and
writes the arrays every other script in this repository starts from.

The label convention is +1 for benign traffic and -1 for attack traffic,
matching the sign of the Pauli-Z expectation value the classifier
measures.

A note on the transform order
-----------------------------
The scaler and PCA are fitted on the full dataset before the train and
test split, so the test features influence the fitted transforms. This is
transductive rather than label leakage, since neither transform sees a
label, and it is disclosed in the paper. It is preserved here rather than
corrected because refitting on the training split alone would produce a
different test set, which would invalidate the hardware runs already
recorded in results/.

Input
-----
data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    Downloaded from https://www.unb.ca/cic/datasets/ids-2017.html and not
    redistributed here.

Outputs
-------
data/X_train.npy, data/y_train.npy    2000 samples
data/X_test.npy,  data/y_test.npy      500 samples

Usage
-----
    python -u 01_preprocess.py
    python -u 01_preprocess.py --csv path/to/capture.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

N_QUBITS = 4
N_TRAIN = 2000
N_TEST = 500
RANDOM_SEED = 42

DEFAULT_CSV = os.path.join(
    "data", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")


def log(message: str) -> None:
    print(message, flush=True)


def load_capture(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    # The published CSVs carry leading spaces in several column names.
    frame.columns = frame.columns.str.strip()
    return frame


def preprocess(frame: pd.DataFrame):
    label_column = next(c for c in frame.columns if "label" in c.lower())
    labels = (frame[label_column].str.strip().str.upper() != "BENIGN")
    labels = labels.astype(int).values

    features = frame.drop(columns=[label_column])
    features = features.apply(pd.to_numeric, errors="coerce")
    # Several flow statistics divide by a duration that can be zero.
    features = features.replace([np.inf, -np.inf], np.nan)
    complete = ~features.isnull().any(axis=1)
    features, labels = features[complete].values, labels[complete]
    log(f"  usable rows: {len(features)}")

    features = MinMaxScaler().fit_transform(features)
    pca = PCA(n_components=N_QUBITS, random_state=RANDOM_SEED)
    features = pca.fit_transform(features)
    log(f"  PCA retains {pca.explained_variance_ratio_.sum():.1%} "
        f"of total variance in {N_QUBITS} components")
    # Angle encoding applies RY(x_i), so the features occupy [0, pi].
    features = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(features)

    # Draw a stratified subsample of the size the experiments use. Taking
    # the second output of the split is how a stratified subsample of a
    # given size is obtained.
    if len(features) > N_TRAIN + N_TEST:
        _, features, _, labels = train_test_split(
            features, labels, test_size=N_TRAIN + N_TEST,
            stratify=labels, random_state=RANDOM_SEED)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=N_TEST, stratify=labels,
        random_state=RANDOM_SEED)

    # Benign maps to +1 and attack to -1, matching the sign convention of
    # the measured expectation value.
    return (X_train, X_test,
            np.where(y_train == 0, 1, -1),
            np.where(y_test == 0, 1, -1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        log(f"{args.csv} not found.")
        log("Download CICIDS2017 from "
            "https://www.unb.ca/cic/datasets/ids-2017.html and place the "
            "Friday afternoon DDoS capture in data/.")
        raise SystemExit(1)

    log(f"reading {args.csv}")
    frame = load_capture(args.csv)
    X_train, X_test, y_train, y_test = preprocess(frame)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, array in (("X_train", X_train), ("X_test", X_test),
                        ("y_train", y_train), ("y_test", y_test)):
        np.save(os.path.join(args.out_dir, f"{name}.npy"), array)

    log(f"\ntrain {len(y_train)}: {(y_train == -1).sum()} attack, "
        f"{(y_train == 1).sum()} benign")
    log(f"test  {len(y_test)}: {(y_test == -1).sum()} attack, "
        f"{(y_test == 1).sum()} benign")
    log(f"feature range [{X_train.min():.3f}, {X_train.max():.3f}]")
    log(f"\nwrote four arrays to {args.out_dir}/")
    log("\nThe published splits have 1134 attack and 866 benign in "
        "training,\nand 284 attack and 216 benign in test. A different "
        "count means a\ndifferent capture file or a different "
        "scikit-learn version.")


if __name__ == "__main__":
    main()
