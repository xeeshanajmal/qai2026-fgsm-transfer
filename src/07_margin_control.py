"""Vary decision margin with everything else held fixed.

The two-model comparison elsewhere in this work differs in optimizer as
well as in decision margin, so it associates margin with adversarial
exposure rather than isolating it. This experiment removes the confound.
A single training run is snapshotted at a sequence of epoch counts, so
every checkpoint shares the same initialization, the same data, the same
optimizer and the same learning rate. Training length is the only thing
that changes, and margin grows with it.

Inputs
------
data/X_train.npy, data/y_train.npy, data/X_test.npy, data/y_test.npy

Output
------
results/margin_control.json
    Per checkpoint and seed: epochs, clean accuracy, decision margin, and
    FGSM success at each budget, with Wilson intervals.

Usage
-----
    python -u 07_margin_control.py
    python -u 07_margin_control.py --seeds 5 --epsilons 0.20 0.30
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

# Identical to the main training script, so checkpoints are comparable with
# the models reported elsewhere.
N_QUBITS = 4
N_LAYERS = 3
N_PARAMS = N_LAYERS * N_QUBITS * 2
LEARNING_RATE = 0.05
BASE_SEED = 1000
VAL_FRACTION = 0.2
FEATURE_RANGE = np.pi

# Checkpoints along the trajectory. Early stopping leaves the classifier
# close to its decision boundary; later checkpoints widen the margin.
CHECKPOINTS = [5, 10, 20, 40, 80, 150]
EPSILONS = [0.20, 0.30]
N_SEEDS = 3
N_ATTACK_SAMPLES = 60


def log(message: str) -> None:
    print(message, flush=True)


device = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(device, interface="autograd", diff_method="backprop")
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
    return np.where(raw_output(X, params) > 0, 1, -1)


def accuracy(X: np.ndarray, y: np.ndarray, params) -> float:
    return float(np.mean(predict(X, params) == y))


def hinge_loss(X, y, params):
    batch = pnp.array(np.atleast_2d(X), requires_grad=False)
    raw = pnp.reshape(circuit(batch, params), (-1,))
    return pnp.mean(pnp.maximum(0.0, 1.0 - pnp.array(y) * raw))


def fgsm(x: np.ndarray, y_true: int, params, epsilon: float) -> np.ndarray:
    sample = pnp.array(np.array(x, dtype=float), requires_grad=True)
    weights = pnp.array(params, requires_grad=False)
    gradient = qml.grad(
        lambda inp: -float(y_true) * circuit(inp, weights))(sample)
    step = epsilon * np.sign(np.array(gradient))
    return np.clip(np.array(x) + step, 0, FEATURE_RANGE)


def wilson_interval(successes: int, trials: int,
                    z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    spread = z * np.sqrt(p * (1 - p) / trials
                         + z * z / (4 * trials * trials)) / denominator
    return float(center - spread), float(center + spread)


def load_splits():
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")

    rng = np.random.default_rng(BASE_SEED)
    validation = []
    for cls in (-1, 1):
        indices = np.where(y_train == cls)[0]
        validation.extend(rng.choice(indices, int(len(indices) * VAL_FRACTION),
                                     replace=False))
    validation = np.array(sorted(validation))
    training = np.setdiff1d(np.arange(len(y_train)), validation)
    return X_train, y_train, training, validation, X_test, y_test


def evaluate(params, X_test, y_test, epsilons, n_samples) -> dict:
    """Clean accuracy, decision margin and attack success for one checkpoint."""
    candidates = np.where(y_test == -1)[0][:n_samples]
    outputs = raw_output(X_test[candidates], params)
    correct = candidates[outputs <= 0]
    margins = np.abs(outputs[outputs <= 0])

    record = {
        "clean_accuracy": round(accuracy(X_test, y_test, params), 4),
        "n_correct_attacks": int(len(correct)),
        "margin_median": round(float(np.median(margins)), 4)
        if len(margins) else 0.0,
        "margin_mean": round(float(margins.mean()), 4) if len(margins) else 0.0,
        "attacks": [],
    }

    for epsilon in epsilons:
        if len(correct) == 0:
            record["attacks"].append({"epsilon": epsilon,
                                      "attack_success_rate": 0.0,
                                      "n_fooled": 0, "n_tested": 0,
                                      "asr_ci95": [0.0, 0.0]})
            continue
        adversarial = np.stack(
            [fgsm(X_test[i], y_test[i], params, epsilon) for i in correct])
        fooled = int(np.sum(predict(adversarial, params) == 1))
        low, high = wilson_interval(fooled, len(correct))
        record["attacks"].append({
            "epsilon": epsilon,
            "attack_success_rate": round(fooled / len(correct), 4),
            "n_fooled": fooled,
            "n_tested": int(len(correct)),
            "asr_ci95": [round(low, 4), round(high, 4)],
        })
    return record


def train_with_checkpoints(X_train, y_train, X_val, y_val, X_test, y_test,
                           seed, checkpoints, epsilons, n_samples) -> list:
    """One trajectory, evaluated at each checkpoint without restarting."""
    rng = np.random.default_rng(seed)
    params = pnp.array(rng.uniform(0, 2 * np.pi, N_PARAMS), requires_grad=True)
    optimizer = qml.AdamOptimizer(stepsize=LEARNING_RATE)

    def loss(p):
        return hinge_loss(X_train, y_train, p)

    records = []
    epoch = 0
    for target in sorted(checkpoints):
        started = time.time()
        while epoch < target:
            params = optimizer.step(loss, params)
            epoch += 1
        snapshot = np.array(params)
        record = evaluate(snapshot, X_test, y_test, epsilons, n_samples)
        record["epochs"] = target
        record["seed"] = seed
        record["validation_accuracy"] = round(
            accuracy(X_val, y_val, snapshot), 4)
        records.append(record)

        rates = "  ".join(
            f"eps {a['epsilon']:.2f}: {a['attack_success_rate']:>6.1%}"
            for a in record["attacks"])
        log(f"    {target:>4} epochs  acc {record['clean_accuracy']:>6.1%}  "
            f"margin {record['margin_median']:.3f}  {rates}  "
            f"({time.time() - started:.0f}s)")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--epsilons", type=float, nargs="*", default=EPSILONS)
    parser.add_argument("--samples", type=int, default=N_ATTACK_SAMPLES)
    args = parser.parse_args()

    X_train, y_train, training, validation, X_test, y_test = load_splits()
    log(f"train {len(training)}  validation {len(validation)}  "
        f"test {len(y_test)}")
    log(f"checkpoints {CHECKPOINTS}, {args.seeds} seeds, "
        f"budgets {args.epsilons}")

    started = time.time()
    all_records = []
    for k in range(args.seeds):
        seed = BASE_SEED + k
        log(f"\n  seed {seed}")
        all_records += train_with_checkpoints(
            X_train[training], y_train[training],
            X_train[validation], y_train[validation],
            X_test, y_test, seed, CHECKPOINTS, args.epsilons, args.samples)

    os.makedirs("results", exist_ok=True)
    with open("results/margin_control.json", "w", encoding="utf-8") as handle:
        json.dump({"checkpoints": CHECKPOINTS, "n_seeds": args.seeds,
                   "epsilons": args.epsilons, "records": all_records},
                  handle, indent=2)
    log("\nsaved results/margin_control.json")

    # Aggregate across seeds, so the summary reads as one curve.
    log("\n" + "=" * 70)
    log("Mean across seeds")
    log("=" * 70)
    header = f"{'epochs':>7} {'accuracy':>10} {'margin':>9}"
    for epsilon in args.epsilons:
        header += f" {'ASR ' + format(epsilon, '.2f'):>11}"
    log(header)

    for target in sorted(CHECKPOINTS):
        rows = [r for r in all_records if r["epochs"] == target]
        line = (f"{target:>7} "
                f"{np.mean([r['clean_accuracy'] for r in rows]):>9.1%} "
                f"{np.mean([r['margin_median'] for r in rows]):>9.3f}")
        for index, epsilon in enumerate(args.epsilons):
            rate = np.mean([r["attacks"][index]["attack_success_rate"]
                            for r in rows])
            line += f" {rate:>10.1%}"
        log(line)

    log(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
