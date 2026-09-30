"""Train the narrow-margin classifier with a gradient-free optimizer.

This produces the second of the two classifiers compared in Section IV-C.
It uses the same circuit, data and loss as the wide-margin model, but
minimizes over random mini-batches with COBYLA rather than over the full
training set with Adam. It reaches comparable clean accuracy and
converges to a solution whose correctly classified samples sit much
closer to the decision boundary.

Reproducibility
---------------
The parameter file published in models/clean_params.npy came from a run
whose initial parameters were drawn from the global NumPy random state
rather than from a seeded generator, so re-running this script produces a
different model with a different margin. The published file is included
in this repository so every number in the paper remains checkable, and
the script seeds its initialization so that runs from here on are
reproducible.

The accuracy printed during training is measured on the test set. It was
recorded for monitoring only: a single run is kept, and no selection is
made among candidates. The wide-margin models in 03_retrain_poisoning_
and_noise.py are selected on a validation split instead.

Inputs
------
data/X_train.npy, data/y_train.npy, data/X_test.npy, data/y_test.npy

Outputs
-------
models/clean_params.npy        trained parameters
results/clean_history.json     final accuracy and the monitoring log

Usage
-----
    python -u 02_train_narrow_margin.py
    python -u 02_train_narrow_margin.py --max-iter 500 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize

# Identical to the wide-margin model, so the two differ only in how they
# are optimized.
N_QUBITS = 4
N_LAYERS = 3
N_PARAMS = N_LAYERS * N_QUBITS * 2

MAX_ITER = 500
BATCH_SIZE = 32
RHOBEG = 0.3
SEED = 42
LOG_EVERY = 50


def log(message: str) -> None:
    print(message, flush=True)


device = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(device, interface="autograd")
def circuit(x, params):
    weights = pnp.reshape(params, (N_LAYERS, N_QUBITS, 2))
    for wire in range(N_QUBITS):
        qml.RY(x[..., wire], wires=wire)
    for layer in range(N_LAYERS):
        for wire in range(N_QUBITS):
            qml.RY(weights[layer, wire, 0], wires=wire)
            qml.RZ(weights[layer, wire, 1], wires=wire)
        for wire in range(N_QUBITS - 1):
            qml.CNOT(wires=[wire, wire + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])
    return qml.expval(qml.PauliZ(0))


def raw_output(X: np.ndarray, params) -> np.ndarray:
    batch = pnp.array(np.atleast_2d(X), requires_grad=False)
    weights = pnp.array(params, requires_grad=False)
    return np.array(pnp.reshape(circuit(batch, weights), (-1,)), dtype=float)


def predict(X: np.ndarray, params) -> np.ndarray:
    """Zero maps to -1, so a sample exactly on the boundary counts as an
    attack. The same rule is used everywhere in this repository."""
    return np.where(raw_output(X, params) > 0, 1, -1)


def accuracy(X: np.ndarray, y: np.ndarray, params) -> float:
    return float(np.mean(predict(X, params) == y))


def hinge_loss(X: np.ndarray, y: np.ndarray, params) -> float:
    return float(np.mean(np.maximum(0.0, 1.0 - y * raw_output(X, params))))


def decision_margin(X: np.ndarray, y: np.ndarray, params) -> float:
    """Median |<Z_0>| over correctly classified attack samples, the
    quantity Section IV-C compares between the two classifiers."""
    attacks = np.where(y == -1)[0]
    outputs = raw_output(X[attacks], params)
    correct = np.abs(outputs[outputs <= 0])
    return float(np.median(correct)) if len(correct) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")
    log(f"train {len(y_train)}  test {len(y_test)}")
    log(f"pennylane {qml.__version__}")

    rng = np.random.default_rng(args.seed)
    params = rng.uniform(0, 2 * np.pi, N_PARAMS)

    history = {"iter": [], "acc": []}
    counter = {"n": 0}

    def objective(candidate):
        # A fresh mini-batch on every evaluation makes the objective
        # stochastic. COBYLA builds a linear model of the objective and
        # assumes it is deterministic, which is why this configuration
        # converges to a narrow-margin solution.
        index = rng.choice(len(X_train), size=args.batch_size, replace=False)
        value = hinge_loss(X_train[index], y_train[index], candidate)

        counter["n"] += 1
        if counter["n"] % LOG_EVERY == 0:
            current = accuracy(X_test, y_test, candidate)
            history["iter"].append(counter["n"])
            history["acc"].append(round(current, 4))
            log(f"  iter {counter['n']:>4}/{args.max_iter}  "
                f"loss {value:.4f}  test acc {current:.1%}")
        return value

    log(f"\ntraining with COBYLA, budget {args.max_iter} iterations")
    started = time.time()
    result = minimize(objective, params, method="COBYLA",
                      options={"maxiter": args.max_iter, "rhobeg": RHOBEG})
    log(f"  finished after {time.time() - started:.0f}s, "
        f"{counter['n']} objective evaluations")

    trained = np.array(result.x)
    final_accuracy = accuracy(X_test, y_test, trained)
    margin = decision_margin(X_test, y_test, trained)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    np.save("models/clean_params.npy", trained)
    with open("results/clean_history.json", "w", encoding="utf-8") as handle:
        json.dump({"final_acc": round(final_accuracy, 4),
                   "decision_margin": round(margin, 4),
                   "max_iter": args.max_iter,
                   "batch_size": args.batch_size,
                   "seed": args.seed,
                   "log": history}, handle)

    log(f"\nclean accuracy  {final_accuracy:.1%}")
    log(f"decision margin {margin:.3f}")
    log("\nsaved models/clean_params.npy and results/clean_history.json")
    log("\nThe published model reaches 86.8% accuracy at a margin of "
        "0.07.\nA different result here is expected: see the note on "
        "reproducibility\nat the top of this file.")


if __name__ == "__main__":
    main()
