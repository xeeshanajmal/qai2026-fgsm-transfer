"""Train the wide-margin models for the poisoning and noise experiments.

Trains the classifier with Adam on the full training batch, at several
label-flipping rates and several depolarizing noise levels. Each
configuration is trained from N_SEEDS initializations, selected on a
validation split carved from the training set, and reported as a mean and
standard deviation across seeds.

The noiseless configuration runs on a statevector simulator
(default.qubit). Every noisy configuration runs on a density matrix
simulator (default.mixed), which is much slower, so the noisy levels
train on a fixed subsample of MAX_TRAIN_NOISY samples. The subsample is a
fixed prefix rather than a random draw, which keeps the objective
deterministic. 06_finish_rq3.py trains a noiseless model at the same size
so the levels can be compared without a training-set-size confound.

Inputs
------
data/X_train.npy, data/y_train.npy, data/X_test.npy, data/y_test.npy

Outputs
-------
results/poisoning_retrained.json
results/noise_retrained.json
models/poisoned_{rate}pct_params.npy   best validation seed
models/noisy_{p}_params.npy            best validation seed

Usage
-----
    python -u 03_retrain_poisoning_and_noise.py
    python -u 03_retrain_poisoning_and_noise.py --smoke
    python -u 03_retrain_poisoning_and_noise.py --skip-poisoning
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

N_QUBITS = 4
N_LAYERS = 3
N_PARAMS = N_LAYERS * N_QUBITS * 2

EPOCHS = 150
LEARNING_RATE = 0.05
N_SEEDS = 3
BASE_SEED = 1000
VAL_FRACTION = 0.2

POISON_RATES = [0.05, 0.10, 0.15]
NOISE_PROBS = [0.0, 0.001, 0.01, 0.05]
POISON_RATE_FOR_NOISE = 0.10
EPSILON_ADV = 0.10
N_FGSM_SAMPLES = 60

# Fixed prefix of the training split used for the density matrix runs.
# A fixed subsample keeps the objective deterministic. Set to None to use
# the full training split, which is far slower.
MAX_TRAIN_NOISY = 400

POISONING_PATH = "results/poisoning_retrained.json"
NOISE_PATH = "results/noise_retrained.json"


def log(*args):
    print(*args, flush=True)


# ----------------------------------------------------------------------------
# Circuit
# ----------------------------------------------------------------------------


def make_circuit(noise_p=0.0):
    """Return a QNode that accepts either one sample or a batch.

    At noise_p = 0 this is statevector simulation on default.qubit. Every
    other level is a density matrix simulation on default.mixed.
    """
    if noise_p == 0.0:
        dev = qml.device("default.qubit", wires=N_QUBITS)
    else:
        dev = qml.device("default.mixed", wires=N_QUBITS)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(x, params):
        w = pnp.reshape(params, (N_LAYERS, N_QUBITS, 2))
        for i in range(N_QUBITS):
            qml.RY(x[..., i], wires=i)
            if noise_p > 0:
                qml.DepolarizingChannel(noise_p, wires=i)
        for layer in range(N_LAYERS):
            for i in range(N_QUBITS):
                qml.RY(w[layer, i, 0], wires=i)
                qml.RZ(w[layer, i, 1], wires=i)
                if noise_p > 0:
                    qml.DepolarizingChannel(noise_p, wires=i)
            for i in range(N_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1])
                if noise_p > 0:
                    qml.DepolarizingChannel(noise_p, wires=i)
                    qml.DepolarizingChannel(noise_p, wires=i + 1)
            qml.CNOT(wires=[N_QUBITS - 1, 0])
            if noise_p > 0:
                qml.DepolarizingChannel(noise_p, wires=N_QUBITS - 1)
                qml.DepolarizingChannel(noise_p, wires=0)
        return qml.expval(qml.PauliZ(0))

    return circuit


# Flipped to False automatically if a device rejects a batched input.
_BROADCAST = {"ok": True}


def outputs(circuit, X, params):
    """Raw expectation values for every row of X, in one call where possible."""
    X = pnp.array(np.atleast_2d(X), requires_grad=False)
    if _BROADCAST["ok"]:
        try:
            return pnp.reshape(circuit(X, params), (-1,))
        except Exception as exc:                     # noqa: BLE001
            _BROADCAST["ok"] = False
            log(f"    [broadcast unavailable, per-sample fallback: {exc}]")
    return pnp.stack([circuit(x, params) for x in X])


def predict(circuit, X, params):
    """Class predictions in {-1, +1}. Zero maps to -1, so ties go to the
    attack class."""
    raw = np.array(outputs(circuit, X, pnp.array(params, requires_grad=False)),
                   dtype=float)
    return np.where(raw > 0, 1, -1)


def accuracy(circuit, X, y, params):
    return float(np.mean(predict(circuit, X, params) == y))


def hinge_loss(circuit, X, y, params):
    raw = outputs(circuit, X, params)
    return pnp.mean(pnp.maximum(0.0, 1.0 - pnp.array(y) * raw))


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------


def train_one_seed(circuit, X_tr, y_tr, X_val, y_val, seed, epochs):
    """Adam on the full training batch. Returns params and validation accuracy."""
    rng = np.random.default_rng(seed)
    params = pnp.array(rng.uniform(0, 2 * np.pi, N_PARAMS), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=LEARNING_RATE)

    def loss_fn(p):
        return hinge_loss(circuit, X_tr, y_tr, p)

    for _ in range(epochs):
        params = opt.step(loss_fn, params)

    return np.array(params), accuracy(circuit, X_val, y_val, params)


def train_multi_seed(circuit, X_tr, y_tr, X_val, y_val, X_te, y_te,
                     label, epochs, n_seeds):
    """Train n_seeds models. Select on validation. Report test spread."""
    best_params, best_val = None, -1.0
    test_accs = []

    for k in range(n_seeds):
        seed = BASE_SEED + k
        t0 = time.time()
        params, val_acc = train_one_seed(circuit, X_tr, y_tr, X_val, y_val,
                                         seed, epochs)
        test_acc = accuracy(circuit, X_te, y_te, params)
        test_accs.append(test_acc)
        log(f"    seed {seed}  val {val_acc:.1%}  test {test_acc:.1%}  "
            f"({time.time() - t0:.0f}s)")
        if val_acc > best_val:
            best_val, best_params = val_acc, params

    selected_test = accuracy(circuit, X_te, y_te, best_params)
    summary = {
        "selected_val_accuracy": round(best_val, 4),
        "selected_test_accuracy": round(selected_test, 4),
        "test_accuracy_mean": round(float(np.mean(test_accs)), 4),
        "test_accuracy_std": round(float(np.std(test_accs)), 4),
        "test_accuracy_all_seeds": [round(a, 4) for a in test_accs],
        "n_seeds": n_seeds,
        "epochs": epochs,
    }
    log(f"  {label}: selected {selected_test:.1%} | "
        f"across seeds {np.mean(test_accs):.1%} +/- {np.std(test_accs):.1%}")
    return best_params, summary


# ----------------------------------------------------------------------------
# Attacks
# ----------------------------------------------------------------------------


def fgsm(circuit, x, y_true, params, epsilon):
    """One gradient-sign step ascending the hinge loss in the input."""
    xi = pnp.array(np.array(x, dtype=float), requires_grad=True)
    p = pnp.array(params, requires_grad=False)
    grad = qml.grad(lambda inp: -float(y_true) * circuit(inp, p))(xi)
    return np.clip(np.array(x) + epsilon * np.sign(np.array(grad)), 0, np.pi)


def fgsm_asr(circuit, X, y, params, epsilon, n_samples):
    """Attack success rate over attack samples the model classifies correctly.

    Samples the model already gets wrong are excluded, since flipping one
    is not an attack success. The same definition is used everywhere in
    this repository.
    """
    idx = np.where(y == -1)[0][:n_samples]
    if len(idx) == 0:
        return 0.0, 0, 0

    preds_orig = predict(circuit, X[idx], params)
    keep = idx[preds_orig == -1]
    if len(keep) == 0:
        return 0.0, 0, 0

    X_adv = np.stack([fgsm(circuit, X[i], y[i], params, epsilon) for i in keep])
    preds_adv = predict(circuit, X_adv, params)
    fooled = int(np.sum(preds_adv == 1))
    return fooled / len(keep), fooled, len(keep)


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(center - half), float(center + half)


def flip_labels(y, rate, seed=42):
    """Label-flipping poisoning: attack labels relabeled benign."""
    rng = np.random.default_rng(seed)
    y_p = y.copy()
    attack_idx = np.where(y == -1)[0]
    n_flip = int(len(attack_idx) * rate)
    y_p[rng.choice(attack_idx, n_flip, replace=False)] = 1
    return y_p, n_flip


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------


def load_splits():
    X_train_full = np.load("data/X_train.npy")
    y_train_full = np.load("data/y_train.npy")
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")

    # Stratified validation split carved from training only. The test set is
    # never used for selection.
    rng = np.random.default_rng(BASE_SEED)
    val_idx = []
    for cls in (-1, 1):
        cls_idx = np.where(y_train_full == cls)[0]
        n_val = int(len(cls_idx) * VAL_FRACTION)
        val_idx.extend(rng.choice(cls_idx, n_val, replace=False))
    val_idx = np.array(sorted(val_idx))
    tr_idx = np.setdiff1d(np.arange(len(y_train_full)), val_idx)

    return X_train_full, y_train_full, tr_idx, val_idx, X_test, y_test


def cap(idx, limit):
    """Fixed prefix of a training index array. Deterministic by construction."""
    return idx if limit is None else idx[:limit]


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------


def run_poisoning(X_full, y_full, tr_idx, val_idx, X_test, y_test, cfg):
    log("\n" + "=" * 60)
    log("Poisoning")
    log("=" * 60)

    circuit = make_circuit(0.0)
    results = []

    for rate in cfg["poison_rates"]:
        log(f"\n  poison rate {rate:.0%}")
        y_poisoned, n_flip = flip_labels(y_full, rate)
        log(f"    flipped {n_flip} attack labels")

        params, summary = train_multi_seed(
            circuit,
            X_full[tr_idx], y_poisoned[tr_idx],
            X_full[val_idx], y_poisoned[val_idx],
            X_test, y_test,
            label=f"poison {rate:.0%}",
            epochs=cfg["epochs"], n_seeds=cfg["n_seeds"],
        )
        np.save(f"models/poisoned_{int(rate * 100)}pct_params.npy", params)

        preds = predict(circuit, X_test, params)
        n_attack = int((y_test == -1).sum())
        n_benign = int((y_test == 1).sum())
        n_missed = int(np.sum(preds[y_test == -1] == 1))
        n_false_alarm = int(np.sum(preds[y_test == 1] == -1))

        entry = {
            "poison_rate": rate,
            "n_flipped": n_flip,
            "clean_accuracy": summary["selected_test_accuracy"],
            "attack_success_rate": round(n_missed / n_attack, 4),
            "asr_ci95": [round(v, 4) for v in wilson(n_missed, n_attack)],
            "false_positive_rate": round(n_false_alarm / n_benign, 4),
            "fpr_ci95": [round(v, 4) for v in wilson(n_false_alarm, n_benign)],
            **summary,
        }
        results.append(entry)
        log(f"    ASR {entry['attack_success_rate']:.1%}  "
            f"FPR {entry['false_positive_rate']:.1%}")

        # Written after every rate, so an interrupted run keeps what it
        # has finished.
        json.dump(results, open(POISONING_PATH, "w"), indent=2)

    log(f"\n  saved {POISONING_PATH}")
    return results


def run_noise(X_full, y_full, tr_idx, val_idx, X_test, y_test, cfg):
    log("\n" + "=" * 60)
    log("Depolarizing noise")
    log("=" * 60)

    y_poisoned, _ = flip_labels(y_full, POISON_RATE_FOR_NOISE)
    results = []

    for noise_p in cfg["noise_probs"]:
        label = "noiseless" if noise_p == 0 else f"p={noise_p}"
        log(f"\n  noise {label}")
        circuit = make_circuit(noise_p)

        tr = tr_idx if noise_p == 0 else cap(tr_idx, MAX_TRAIN_NOISY)
        if len(tr) != len(tr_idx):
            log(f"    training on a fixed subsample of {len(tr)}")

        params_clean, summary_clean = train_multi_seed(
            circuit,
            X_full[tr], y_full[tr],
            X_full[val_idx], y_full[val_idx],
            X_test, y_test,
            label=f"{label} clean",
            epochs=cfg["epochs"], n_seeds=cfg["n_seeds"],
        )
        np.save(f"models/noisy_{noise_p}_params.npy", params_clean)

        params_poison, _ = train_multi_seed(
            circuit,
            X_full[tr], y_poisoned[tr],
            X_full[val_idx], y_poisoned[val_idx],
            X_test, y_test,
            label=f"{label} poisoned",
            epochs=cfg["epochs"], n_seeds=cfg["n_seeds"],
        )

        preds_p = predict(circuit, X_test, params_poison)
        n_attack = int((y_test == -1).sum())
        n_missed = int(np.sum(preds_p[y_test == -1] == 1))

        asr, fooled, tested = fgsm_asr(
            circuit, X_test, y_test, params_clean,
            EPSILON_ADV, cfg["n_fgsm_samples"],
        )

        entry = {
            "noise_prob": noise_p,
            "label": label,
            "simulator": "statevector" if noise_p == 0 else "density matrix",
            "n_train": int(len(tr)),
            "clean_accuracy": summary_clean["selected_test_accuracy"],
            "clean_accuracy_mean": summary_clean["test_accuracy_mean"],
            "clean_accuracy_std": summary_clean["test_accuracy_std"],
            "poison_asr": round(n_missed / n_attack, 4),
            "poison_asr_ci95": [round(v, 4) for v in wilson(n_missed, n_attack)],
            "fgsm_asr": round(asr, 4),
            "fgsm_asr_ci95": [round(v, 4) for v in wilson(fooled, tested)],
            "fgsm_n_tested": tested,
            "fgsm_n_fooled": fooled,
        }
        results.append(entry)
        log(f"    clean {entry['clean_accuracy']:.1%} | "
            f"poison ASR {entry['poison_asr']:.1%} | "
            f"FGSM ASR {asr:.1%} ({fooled}/{tested})")

        json.dump(results, open(NOISE_PATH, "w"), indent=2)

    log(f"\n  saved {NOISE_PATH}")
    return results


# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true",
                        help="fast end-to-end check, results not for the paper")
    parser.add_argument("--skip-poisoning", action="store_true",
                        help="run only the noise stage and read the poisoning "
                             "results from disk")
    args = parser.parse_args()

    if args.smoke:
        cfg = {"epochs": 20, "n_seeds": 2, "poison_rates": [0.10],
               "noise_probs": [0.0, 0.01], "n_fgsm_samples": 10}
        log("Smoke test. The numbers below are not usable in the paper.\n")
    else:
        cfg = {"epochs": EPOCHS, "n_seeds": N_SEEDS,
               "poison_rates": POISON_RATES, "noise_probs": NOISE_PROBS,
               "n_fgsm_samples": N_FGSM_SAMPLES}

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    for path in ("data/X_train.npy", "data/y_train.npy",
                 "data/X_test.npy", "data/y_test.npy"):
        if not os.path.exists(path):
            log(f"missing {path}. Run this from the folder that holds data/.")
            sys.exit(1)

    if args.skip_poisoning and not os.path.exists(POISONING_PATH):
        log(f"--skip-poisoning needs {POISONING_PATH}, which does not exist.")
        log("Run without the flag first.")
        sys.exit(1)

    t_start = time.time()
    X_full, y_full, tr_idx, val_idx, X_test, y_test = load_splits()
    log(f"train {len(tr_idx)}  validation {len(val_idx)}  test {len(y_test)}")
    log(f"pennylane {qml.__version__}")

    if args.skip_poisoning:
        log(f"\nreading poisoning results from {POISONING_PATH}")
        poisoning = json.load(open(POISONING_PATH))
    else:
        poisoning = run_poisoning(X_full, y_full, tr_idx, val_idx,
                                  X_test, y_test, cfg)

    noise = run_noise(X_full, y_full, tr_idx, val_idx, X_test, y_test, cfg)

    log("\n" + "=" * 70)
    log("Poisoning summary")
    log("=" * 70)
    log(f"{'rate':>6} | {'clean acc across seeds':>24} | {'ASR':>8} | {'FPR':>8}")
    for r in poisoning:
        spread = f"{r['test_accuracy_mean']:.1%} +/- {r['test_accuracy_std']:.1%}"
        log(f"{r['poison_rate']:>5.0%} | {spread:>24} | "
            f"{r['attack_success_rate']:>7.1%} | {r['false_positive_rate']:>7.1%}")

    log("\n" + "=" * 70)
    log("Noise summary")
    log("=" * 70)
    log(f"{'level':>10} | {'clean acc across seeds':>24} | "
        f"{'poison ASR':>10} | {'FGSM ASR':>9}")
    for r in noise:
        spread = f"{r['clean_accuracy_mean']:.1%} +/- {r['clean_accuracy_std']:.1%}"
        log(f"{r['label']:>10} | {spread:>24} | "
            f"{r['poison_asr']:>9.1%} | {r['fgsm_asr']:>8.1%}")

    log(f"\ntotal {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
