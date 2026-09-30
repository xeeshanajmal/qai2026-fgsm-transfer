"""Compare gradient-guided against untargeted perturbation.

FGSM uses the loss gradient to choose the direction of each perturbation.
The untargeted baseline applies uniform random noise within the same
budget to the same quantities: because the encoding applies RY(x_i)
directly, the rotation angles and the preprocessed input features are
identical, so both attacks perturb exactly the same vector. The only
difference is whether the gradient guides the step, which makes the
comparison a measurement of what gradient access is worth to an attacker.

Both the wide-margin and narrow-margin models are evaluated, so the
comparison can be read alongside the margin results.

The untargeted baseline is averaged over several random draws, since a
single draw at a given budget is itself a random quantity.

Inputs
------
data/X_test.npy, data/y_test.npy
models/noisy_0.0_params.npy          wide-margin model
models/clean_params.npy              narrow-margin model

Output
------
results/gradient_vs_untargeted.json

Usage
-----
    python -u 05_compare_attack_guidance.py
    python -u 05_compare_attack_guidance.py --trials 20 --n 60
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

N_QUBITS = 4
N_LAYERS = 3
FEATURE_RANGE = np.pi

EPSILONS = [0.05, 0.10, 0.20, 0.30, 0.50]
N_SAMPLES = 60
N_TRIALS = 10
SEED = 1000

MODELS = [
    ("wide_margin", "models/noisy_0.0_params.npy"),
    ("narrow_margin", "models/clean_params.npy"),
]


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


def raw_output(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    batch = pnp.array(np.atleast_2d(X), requires_grad=False)
    weights = pnp.array(params, requires_grad=False)
    return np.array(pnp.reshape(circuit(batch, weights), (-1,)), dtype=float)


def predict(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    return np.where(raw_output(X, params) > 0, 1, -1)


def fgsm(x: np.ndarray, y_true: int, params: np.ndarray,
         epsilon: float) -> np.ndarray:
    sample = pnp.array(np.array(x, dtype=float), requires_grad=True)
    weights = pnp.array(params, requires_grad=False)
    gradient = qml.grad(
        lambda inp: -float(y_true) * circuit(inp, weights))(sample)
    step = epsilon * np.sign(np.array(gradient))
    return np.clip(np.array(x) + step, 0, FEATURE_RANGE)


def untargeted(x: np.ndarray, epsilon: float,
               rng: np.random.Generator) -> np.ndarray:
    """Uniform noise within the same budget, applied to the same vector."""
    step = rng.uniform(-epsilon, epsilon, N_QUBITS)
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


def evaluate(name: str, params: np.ndarray, X_test: np.ndarray,
             y_test: np.ndarray, n_samples: int, n_trials: int) -> dict:
    log("\n" + "=" * 64)
    log(f"MODEL: {name}")
    log("=" * 64)

    candidates = np.where(y_test == -1)[0][:n_samples]
    outputs = raw_output(X_test[candidates], params)
    correct = candidates[outputs <= 0]
    margins = np.abs(outputs[outputs <= 0])

    log(f"  correctly classified attack samples: "
        f"{len(correct)}/{len(candidates)}")
    if len(correct) == 0:
        return {"model": name, "budgets": []}
    log(f"  median decision margin: {np.median(margins):.3f}")

    rng = np.random.default_rng(SEED)
    rows = []

    log(f"\n  {'eps':>6} {'% range':>8} {'FGSM':>16} {'untargeted':>18}")
    log("  " + "-" * 54)

    for epsilon in EPSILONS:
        adversarial = np.stack(
            [fgsm(X_test[i], y_test[i], params, epsilon) for i in correct])
        fooled = int(np.sum(predict(adversarial, params) == 1))
        fgsm_rate = fooled / len(correct)
        fgsm_ci = wilson_interval(fooled, len(correct))

        # Each trial is one independent draw of noise over all samples.
        trial_rates = []
        for _ in range(n_trials):
            perturbed = np.stack(
                [untargeted(X_test[i], epsilon, rng) for i in correct])
            trial_rates.append(
                float(np.mean(predict(perturbed, params) == 1)))

        untargeted_mean = float(np.mean(trial_rates))
        untargeted_sd = float(np.std(trial_rates))

        ratio = (fgsm_rate / untargeted_mean) if untargeted_mean > 0 else None
        ratio_text = f"{ratio:.1f}x" if ratio else "n/a"

        log(f"  {epsilon:>6.2f} {100 * epsilon / FEATURE_RANGE:>7.1f}% "
            f"{fgsm_rate:>7.1%} {fooled:>3}/{len(correct):<3} "
            f"{untargeted_mean:>9.1%} +/- {untargeted_sd:<5.1%} {ratio_text:>6}")

        rows.append({
            "epsilon": epsilon,
            "epsilon_as_fraction_of_range": round(epsilon / FEATURE_RANGE, 4),
            "fgsm_asr": round(fgsm_rate, 4),
            "fgsm_n_fooled": fooled,
            "fgsm_n_tested": int(len(correct)),
            "fgsm_ci95": [round(b, 4) for b in fgsm_ci],
            "untargeted_asr_mean": round(untargeted_mean, 4),
            "untargeted_asr_std": round(untargeted_sd, 4),
            "untargeted_n_trials": n_trials,
            "guidance_ratio": round(ratio, 2) if ratio else None,
        })

    return {
        "model": name,
        "n_correct": int(len(correct)),
        "margin_median": round(float(np.median(margins)), 4),
        "budgets": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=N_SAMPLES)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")
    log(f"test set {len(y_test)}, features scaled to "
        f"[0, {FEATURE_RANGE:.3f}]")
    log(f"untargeted baseline averaged over {args.trials} random draws")

    results = []
    for name, path in MODELS:
        if not os.path.exists(path):
            log(f"\n  {path} not found; skipping {name}")
            continue
        results.append(evaluate(name, np.load(path), X_test, y_test,
                                args.n, args.trials))

    os.makedirs("results", exist_ok=True)
    with open("results/gradient_vs_untargeted.json", "w",
              encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    log("\nsaved results/gradient_vs_untargeted.json")


if __name__ == "__main__":
    main()
