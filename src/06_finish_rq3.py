"""
Two gaps left in RQ3 after the main retraining run.

1. The noiseless level trained on 1601 samples while the noisy levels trained
   on a fixed subsample of 400. Comparing them across noise levels is
   confounded by training set size. This retrains the noiseless level on the
   same 400 samples so there is a like-for-like row.

2. The FGSM column came out 0.0% at every noise level because it ran at
   epsilon 0.10, where the epsilon sweep already showed the attack does
   nothing against an Adam-trained model. Re-evaluating at epsilon 0.30 and
   0.50 turns that column from "no signal" into an answer about whether
   simulated device noise changes attack success.

Part 2 is evaluation only. It loads the parameter files the main run already
saved and never retrains, so it costs seconds per noise level rather than the
16 minutes per seed the training took.

Inputs  : data/X_train.npy, data/y_train.npy, data/X_test.npy, data/y_test.npy
          models/noisy_0.0_params.npy, models/noisy_0.001_params.npy,
          models/noisy_0.01_params.npy, models/noisy_0.05_params.npy
Outputs : results/noiseless_400_baseline.json
          results/noise_fgsm_sweep.json

Usage:
    python -u finish_rq3.py                 # both parts
    python -u finish_rq3.py --only fgsm     # part 2 only, seconds
    python -u finish_rq3.py --only baseline # part 1 only, ~2 minutes
"""

import argparse
import json
import os
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

# Must match retrain_poisoning_and_noise.py exactly.
N_QUBITS = 4
N_LAYERS = 3
N_PARAMS = N_LAYERS * N_QUBITS * 2
EPOCHS = 150
LEARNING_RATE = 0.05
N_SEEDS = 3
BASE_SEED = 1000
VAL_FRACTION = 0.2
MAX_TRAIN_NOISY = 400

NOISE_PROBS = [0.0, 0.001, 0.01, 0.05]
FGSM_EPSILONS = [0.30, 0.50]
N_FGSM_SAMPLES = 60
FEATURE_RANGE = np.pi


def log(*args):
    print(*args, flush=True)


# ----------------------------------------------------------------------------
# Circuit, identical to the training script
# ----------------------------------------------------------------------------


def make_circuit(noise_p=0.0):
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


_BROADCAST = {"ok": True}


def outputs(circuit, X, params):
    X = pnp.array(np.atleast_2d(X), requires_grad=False)
    if _BROADCAST["ok"]:
        try:
            return pnp.reshape(circuit(X, params), (-1,))
        except Exception as exc:                     # noqa: BLE001
            _BROADCAST["ok"] = False
            log(f"    [broadcast unavailable, per-sample fallback: {exc}]")
    return pnp.stack([circuit(x, params) for x in X])


def predict(circuit, X, params):
    raw = np.array(outputs(circuit, X, pnp.array(params, requires_grad=False)),
                   dtype=float)
    return np.where(raw > 0, 1, -1)


def raw_out(circuit, X, params):
    return np.array(outputs(circuit, X, pnp.array(params, requires_grad=False)),
                    dtype=float)


def accuracy(circuit, X, y, params):
    return float(np.mean(predict(circuit, X, params) == y))


def hinge_loss(circuit, X, y, params):
    raw = outputs(circuit, X, params)
    return pnp.mean(pnp.maximum(0.0, 1.0 - pnp.array(y) * raw))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(centre - half), float(centre + half)


def flip_labels(y, rate, seed=42):
    rng = np.random.default_rng(seed)
    y_p = y.copy()
    attack_idx = np.where(y == -1)[0]
    n_flip = int(len(attack_idx) * rate)
    y_p[rng.choice(attack_idx, n_flip, replace=False)] = 1
    return y_p, n_flip


def load_splits():
    X_train_full = np.load("data/X_train.npy")
    y_train_full = np.load("data/y_train.npy")
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")

    rng = np.random.default_rng(BASE_SEED)
    val_idx = []
    for cls in (-1, 1):
        cls_idx = np.where(y_train_full == cls)[0]
        n_val = int(len(cls_idx) * VAL_FRACTION)
        val_idx.extend(rng.choice(cls_idx, n_val, replace=False))
    val_idx = np.array(sorted(val_idx))
    tr_idx = np.setdiff1d(np.arange(len(y_train_full)), val_idx)
    return X_train_full, y_train_full, tr_idx, val_idx, X_test, y_test


# ----------------------------------------------------------------------------
# Part 1: like-for-like noiseless baseline
# ----------------------------------------------------------------------------


def train_one_seed(circuit, X_tr, y_tr, X_val, y_val, seed):
    rng = np.random.default_rng(seed)
    params = pnp.array(rng.uniform(0, 2 * np.pi, N_PARAMS), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=LEARNING_RATE)

    def loss_fn(p):
        return hinge_loss(circuit, X_tr, y_tr, p)

    for _ in range(EPOCHS):
        params = opt.step(loss_fn, params)
    return np.array(params), accuracy(circuit, X_val, y_val, params)


def run_baseline(X_full, y_full, tr_idx, val_idx, X_test, y_test):
    log("\n" + "=" * 62)
    log("PART 1: noiseless baseline on the same 400 samples as the noisy levels")
    log("=" * 62)

    circuit = make_circuit(0.0)
    tr = tr_idx[:MAX_TRAIN_NOISY]
    y_poisoned, _ = flip_labels(y_full, 0.10)
    log(f"training on {len(tr)} samples, {N_SEEDS} seeds")

    out = {"n_train": int(len(tr)), "epochs": EPOCHS, "n_seeds": N_SEEDS}

    for tag, y_used in (("clean", y_full), ("poisoned", y_poisoned)):
        log(f"\n  {tag}")
        best_params, best_val, accs = None, -1.0, []
        for k in range(N_SEEDS):
            seed = BASE_SEED + k
            t0 = time.time()
            params, val_acc = train_one_seed(circuit, X_full[tr], y_used[tr],
                                             X_full[val_idx], y_used[val_idx],
                                             seed)
            test_acc = accuracy(circuit, X_test, y_test, params)
            accs.append(test_acc)
            log(f"    seed {seed}  val {val_acc:.1%}  test {test_acc:.1%}  "
                f"({time.time() - t0:.0f}s)")
            if val_acc > best_val:
                best_val, best_params = val_acc, params

        if tag == "clean":
            np.save("models/noisy_0.0_params_400.npy", best_params)

        out[tag] = {
            "selected_test_accuracy": round(
                accuracy(circuit, X_test, y_test, best_params), 4),
            "test_accuracy_mean": round(float(np.mean(accs)), 4),
            "test_accuracy_std": round(float(np.std(accs)), 4),
            "test_accuracy_all_seeds": [round(a, 4) for a in accs],
        }
        log(f"  {tag}: selected {out[tag]['selected_test_accuracy']:.1%} | "
            f"across seeds {np.mean(accs):.1%} +/- {np.std(accs):.1%}")

        if tag == "poisoned":
            preds = predict(circuit, X_test, best_params)
            n_attack = int((y_test == -1).sum())
            n_missed = int(np.sum(preds[y_test == -1] == 1))
            out["poison_asr"] = round(n_missed / n_attack, 4)
            out["poison_asr_ci95"] = [round(v, 4)
                                      for v in wilson(n_missed, n_attack)]
            log(f"  poison ASR {out['poison_asr']:.1%}")

    json.dump(out, open("results/noiseless_400_baseline.json", "w"), indent=2)
    log("\n  saved results/noiseless_400_baseline.json")
    return out


# ----------------------------------------------------------------------------
# Part 2: FGSM at working epsilon, across noise levels
# ----------------------------------------------------------------------------


def fgsm(circuit, x, y_true, params, epsilon):
    xi = pnp.array(np.array(x, dtype=float), requires_grad=True)
    p = pnp.array(params, requires_grad=False)
    grad = qml.grad(lambda inp: -float(y_true) * circuit(inp, p))(xi)
    return np.clip(np.array(x) + epsilon * np.sign(np.array(grad)),
                   0, FEATURE_RANGE)


def run_fgsm(X_test, y_test):
    log("\n" + "=" * 62)
    log("PART 2: FGSM across noise levels at working epsilon")
    log("=" * 62)
    log("Attacks are generated against each noise level's own model, using")
    log("that level's simulator. This asks whether simulated device noise")
    log("changes how vulnerable the model is, which is what RQ3 should test.")

    results = []
    for noise_p in NOISE_PROBS:
        path = f"models/noisy_{noise_p}_params.npy"
        if not os.path.exists(path):
            log(f"\n  skipping p={noise_p}: {path} not found")
            continue

        label = "noiseless" if noise_p == 0 else f"p={noise_p}"
        log(f"\n  {label}")
        circuit = make_circuit(noise_p)
        params = np.load(path)

        attack_idx = np.where(y_test == -1)[0][:N_FGSM_SAMPLES]
        raw = raw_out(circuit, X_test[attack_idx], params)
        keep = attack_idx[raw <= 0]
        margins = np.abs(raw[raw <= 0])

        log(f"    correctly classified: {len(keep)}/{len(attack_idx)}")
        if len(keep) == 0:
            log("    nothing to attack")
            continue
        log(f"    median |output|: {np.median(margins):.3f}")

        entry = {
            "noise_prob": noise_p,
            "label": label,
            "n_correct": int(len(keep)),
            "margin_median": round(float(np.median(margins)), 4),
            "margin_mean": round(float(margins.mean()), 4),
            "epsilons": [],
        }

        for eps in FGSM_EPSILONS:
            t0 = time.time()
            X_adv = np.stack([fgsm(circuit, X_test[i], y_test[i], params, eps)
                              for i in keep])
            fooled = int(np.sum(predict(circuit, X_adv, params) == 1))
            asr = fooled / len(keep)
            lo, hi = wilson(fooled, len(keep))
            log(f"    eps {eps:.2f} ({100 * eps / FEATURE_RANGE:.1f}% range)  "
                f"ASR {asr:>6.1%}  {fooled}/{len(keep)}  "
                f"[{lo:.2f},{hi:.2f}]  ({time.time() - t0:.0f}s)")
            entry["epsilons"].append({
                "epsilon": eps,
                "attack_success_rate": round(asr, 4),
                "n_fooled": fooled,
                "n_tested": int(len(keep)),
                "asr_ci95": [round(lo, 4), round(hi, 4)],
            })

        results.append(entry)

    json.dump(results, open("results/noise_fgsm_sweep.json", "w"), indent=2)
    log("\n  saved results/noise_fgsm_sweep.json")

    if results:
        log("\n" + "=" * 62)
        log("SUMMARY")
        log("=" * 62)
        header = f"{'level':>10} {'margin':>8}"
        for eps in FGSM_EPSILONS:
            header += f" {'eps ' + format(eps, '.2f'):>10}"
        log(header)
        for r in results:
            row = f"{r['label']:>10} {r['margin_median']:>8.3f}"
            for e in r["epsilons"]:
                row += f" {e['attack_success_rate']:>9.1%}"
            log(row)
        log("\nIf ASR is flat across noise levels, simulated depolarizing noise")
        log("does not change adversarial vulnerability. That matches hardware")
        log("classifying 50/50 clean samples correctly.")

    return results


# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["baseline", "fgsm"], default=None)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X_full, y_full, tr_idx, val_idx, X_test, y_test = load_splits()
    log(f"train {len(tr_idx)}  validation {len(val_idx)}  test {len(y_test)}")
    log(f"pennylane {qml.__version__}")

    t0 = time.time()
    if args.only in (None, "baseline"):
        run_baseline(X_full, y_full, tr_idx, val_idx, X_test, y_test)
    if args.only in (None, "fgsm"):
        run_fgsm(X_test, y_test)
    log(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
