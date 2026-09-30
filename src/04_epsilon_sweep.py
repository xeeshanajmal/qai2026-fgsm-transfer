"""FGSM success against perturbation budget, for both classifiers.

Sweeps epsilon against the narrow-margin model (COBYLA) and the
wide-margin model (Adam), and records attack success at each budget with
Wilson intervals. It also reports each model's decision margin, measured
as the median |<Z_0>| over correctly classified attack samples, which is
the quantity that differs between them.

Features are scaled to [0, pi], so epsilon 0.31 is 10% of the feature
range and epsilon 0.63 is 20%. The script prints that fraction beside
every epsilon so the perturbation size stays interpretable.

Inputs
------
data/X_test.npy, data/y_test.npy
models/clean_params.npy          narrow margin, COBYLA
models/noisy_0.0_params.npy      wide margin, Adam

Output
------
results/epsilon_sweep.json

Usage
-----
    python -u 04_epsilon_sweep.py
    python -u 04_epsilon_sweep.py --n 100
    python -u 04_epsilon_sweep.py --model adam
"""

import argparse
import json
import os
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

N_QUBITS = 4
N_LAYERS = 3
FEATURE_RANGE = np.pi

EPSILONS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]
DEFAULT_N = 60


def log(*args):
    print(*args, flush=True)


dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="autograd", diff_method="backprop")
def circuit(x, params):
    w = pnp.reshape(params, (N_LAYERS, N_QUBITS, 2))
    for i in range(N_QUBITS):
        qml.RY(x[..., i], wires=i)
    for layer in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(w[layer, i, 0], wires=i)
            qml.RZ(w[layer, i, 1], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])
    return qml.expval(qml.PauliZ(0))


def raw_outputs(X, params):
    X = pnp.array(np.atleast_2d(X), requires_grad=False)
    p = pnp.array(params, requires_grad=False)
    return np.array(pnp.reshape(circuit(X, p), (-1,)), dtype=float)


def predict(X, params):
    return np.where(raw_outputs(X, params) > 0, 1, -1)


def fgsm(x, y_true, params, epsilon):
    """One gradient-sign step ascending the hinge loss in the input."""
    xi = pnp.array(np.array(x, dtype=float), requires_grad=True)
    p = pnp.array(params, requires_grad=False)
    grad = qml.grad(lambda inp: -float(y_true) * circuit(inp, p))(xi)
    return np.clip(np.array(x) + epsilon * np.sign(np.array(grad)), 0, FEATURE_RANGE)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(center - half), float(center + half)


def sweep(name, params, X_test, y_test, n_samples):
    """Sweep epsilon against one model. Returns a result dict."""
    log("\n" + "=" * 62)
    log(f"MODEL: {name}")
    log("=" * 62)

    acc = float(np.mean(predict(X_test, params) == y_test))
    log(f"  clean test accuracy: {acc:.1%}")

    # |expectation| on attack samples the model gets right says how far
    # FGSM has to move the output before a decision flips.
    attack_idx = np.where(y_test == -1)[0][:n_samples]
    raw = raw_outputs(X_test[attack_idx], params)
    correct = raw <= 0
    keep = attack_idx[correct]
    margins = np.abs(raw[correct])

    log(f"  attack samples used: {len(keep)} of {len(attack_idx)}")
    if len(margins):
        log(f"  |output| on correct attacks: median {np.median(margins):.3f}  "
            f"mean {margins.mean():.3f}  max {margins.max():.3f}")
    else:
        log("  no correctly classified attack samples, nothing to attack")
        return {"model": name, "clean_accuracy": round(acc, 4), "sweep": []}

    rows = []
    log(f"\n  {'eps':>6} {'% range':>8} {'ASR':>8} {'fooled':>8} "
        f"{'95% CI':>16} {'mean |dx|':>10}")
    log("  " + "-" * 60)

    for eps in EPSILONS:
        t0 = time.time()
        X_adv = np.stack([fgsm(X_test[i], y_test[i], params, eps) for i in keep])
        preds_adv = predict(X_adv, params)
        fooled = int(np.sum(preds_adv == 1))
        asr = fooled / len(keep)
        lo, hi = wilson(fooled, len(keep))
        shift = float(np.mean(np.linalg.norm(X_adv - X_test[keep], axis=1)))

        log(f"  {eps:>6.2f} {100 * eps / FEATURE_RANGE:>7.1f}% {asr:>7.1%} "
            f"{fooled:>4}/{len(keep):<3} [{lo:>5.2f},{hi:>5.2f}] {shift:>10.3f}"
            f"   ({time.time() - t0:.0f}s)")

        rows.append({
            "epsilon": eps,
            "epsilon_as_fraction_of_range": round(eps / FEATURE_RANGE, 4),
            "attack_success_rate": round(asr, 4),
            "n_fooled": fooled,
            "n_tested": int(len(keep)),
            "asr_ci95": [round(lo, 4), round(hi, 4)],
            "mean_l2_perturbation": round(shift, 4),
        })

    return {
        "model": name,
        "clean_accuracy": round(acc, 4),
        "n_attack_samples_correct": int(len(keep)),
        "margin_median": round(float(np.median(margins)), 4),
        "margin_mean": round(float(margins.mean()), 4),
        "sweep": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help="attack samples to test per epsilon")
    parser.add_argument("--model", choices=["both", "cobyla", "adam"],
                        default="both")
    args = parser.parse_args()

    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")
    log(f"test set {len(y_test)}  attacks {(y_test == -1).sum()}  "
        f"benign {(y_test == 1).sum()}")
    log(f"features scaled to [0, {FEATURE_RANGE:.3f}]")

    wanted = []
    if args.model in ("both", "cobyla"):
        wanted.append(("COBYLA (models/clean_params.npy)", "models/clean_params.npy"))
    if args.model in ("both", "adam"):
        wanted.append(("Adam (models/noisy_0.0_params.npy)", "models/noisy_0.0_params.npy"))

    results = []
    for name, path in wanted:
        if not os.path.exists(path):
            log(f"\nskipping {name}: {path} not found")
            continue
        results.append(sweep(name, np.load(path), X_test, y_test, args.n))

    os.makedirs("results", exist_ok=True)
    json.dump(results, open("results/epsilon_sweep.json", "w"), indent=2)
    log("\nsaved results/epsilon_sweep.json")
    if args.model != "both":
        log("--model writes a file holding only that model. Run without it "
            "to restore both entries.")


if __name__ == "__main__":
    main()
