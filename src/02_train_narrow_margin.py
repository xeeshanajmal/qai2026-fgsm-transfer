"""Train the narrow-margin classifier with a gradient-free optimizer.

This produces the second of the two classifiers compared in Section IV-C.
It uses the same circuit, data and loss as the wide-margin model, but
minimizes over random mini-batches with COBYLA rather than over the full
training set with Adam. It reaches comparable clean accuracy and
converges to a solution whose correctly classified samples sit much
closer to the decision boundary.

Reproducibility
---------------
The parameter file in models/clean_params.npy came from a run whose
initial parameters were drawn from the global NumPy random state rather
than from a seeded generator, so this script cannot reproduce it. The
file is included so that every number in the paper stays checkable, and
the script refuses to overwrite it without --force.

This script seeds its initialization, so runs from here on are
reproducible among themselves. They land on different solutions, and the
spread is wide: a gradient-free optimizer on a stochastic objective can
converge to a model no better than predicting the majority class.

The accuracy printed during training is measured on the test set. It was
recorded for monitoring only, and no selection is made among candidates.
The wide-margin models in 03_retrain_poisoning_and_noise.py are selected
on a validation split instead.

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

TARGET = "models/clean_params.npy"


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


def decision_margin(X: np.ndarray, y: np.ndarray,
                    params) -> tuple[float, int]:
    """Median |<Z_0>| over correctly classified attack samples, with the
    number of samples it was computed over.

    The count matters. A model that classifies few attacks correctly
    yields a median over a handful of values, which looks like a normal
    margin but carries no information.
    """
    attacks = np.where(y == -1)[0]
    outputs = raw_output(X[attacks], params)
    correct = np.abs(outputs[outputs <= 0])
    if len(correct) == 0:
        return 0.0, 0
    return float(np.median(correct)), int(len(correct))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing parameter file")
    args = parser.parse_args()

    # The parameters used in the paper cannot be regenerated, so
    # overwriting them loses them permanently.
    if os.path.exists(TARGET) and not args.force:
        log(f"{TARGET} already exists and would be overwritten.")
        log("It cannot be regenerated: see the note at the top of this "
            "file.")
        log("Pass --force to overwrite it.")
        raise SystemExit(1)

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
    margin, margin_samples = decision_margin(X_test, y_test, trained)

    # A classifier that predicts one class for everything scores the
    # majority-class fraction. Falling near or below it means training
    # collapsed rather than converged.
    majority = max((y_test == -1).mean(), (y_test == 1).mean())

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    np.save(TARGET, trained)
    with open("results/clean_history.json", "w", encoding="utf-8") as handle:
        json.dump({"final_acc": round(final_accuracy, 4),
                   "decision_margin": round(margin, 4),
                   "margin_n_samples": margin_samples,
                   "max_iter": args.max_iter,
                   "batch_size": args.batch_size,
                   "seed": args.seed,
                   "log": history}, handle)

    log(f"\nclean accuracy  {final_accuracy:.1%}  "
        f"(majority class {majority:.1%})")
    log(f"decision margin {margin:.3f} over {margin_samples} correctly "
        f"classified attack samples")
    log(f"\nsaved {TARGET} and results/clean_history.json")

    if final_accuracy <= majority + 0.02:
        log("\nThis run did not converge: accuracy is at or below the "
            "majority-class\nbaseline, so the model predicts one class "
            "for nearly everything. The\nmargin above is computed over "
            "too few samples to mean anything.")


if __name__ == "__main__":
    main()
