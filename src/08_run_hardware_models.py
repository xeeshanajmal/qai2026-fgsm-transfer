"""Evaluate trained classifiers on IBM Quantum hardware.

Measures classification accuracy of one or more trained models on
unperturbed test samples, so that device behavior can be compared against
simulation. Every circuit is submitted as a single job, so all models are
evaluated under the same calibration state and the comparison between
them is not confounded by device drift.

Adversarial transfer is measured separately by 09_run_hardware_fgsm.py.

Credentials are read from the environment:

    PowerShell:
        $env:IBM_API_KEY = "<token>"
        $env:IBM_INSTANCE = "<crn>"

    cmd:
        set IBM_API_KEY=<token>
        set IBM_INSTANCE=<crn>

Inputs
------
data/X_test.npy, data/y_test.npy
models/noisy_0.0_params.npy              clean model
models/poisoned_10pct_params.npy         poisoned model

Output
------
results/hardware_clean_vs_poisoned.json
    Simulator and hardware accuracy for each model, with Wilson
    intervals, per-sample predictions and transpilation statistics.

Usage
-----
    python -u 08_run_hardware_models.py
    python -u 08_run_hardware_models.py --dry-run
    python -u 08_run_hardware_models.py --n 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

N_QUBITS = 4
N_LAYERS = 3
N_SHOTS = 1024
N_SAMPLES = 50
BACKEND_NAME = "ibm_fez"
OPTIMIZATION_LEVEL = 1

MODELS = [
    ("clean", "models/noisy_0.0_params.npy"),
    ("poisoned_10pct", "models/poisoned_10pct_params.npy"),
]


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Simulator reference
# ---------------------------------------------------------------------------

_simulator = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(_simulator, interface="autograd")
def _circuit(x, params):
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


def simulator_predict(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Class predictions in {-1, +1}. Zero maps to -1, matching the
    hardware tie-breaking rule below."""
    batch = pnp.array(np.atleast_2d(X), requires_grad=False)
    weights = pnp.array(params, requires_grad=False)
    raw = np.array(pnp.reshape(_circuit(batch, weights), (-1,)), dtype=float)
    return np.where(raw > 0, 1, -1)


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------


def build_circuit(x: np.ndarray, params: np.ndarray) -> QuantumCircuit:
    weights = np.asarray(params).reshape(N_LAYERS, N_QUBITS, 2)
    circuit = QuantumCircuit(N_QUBITS, 1)
    for wire in range(N_QUBITS):
        circuit.ry(float(x[wire]), wire)
    for layer in range(N_LAYERS):
        for wire in range(N_QUBITS):
            circuit.ry(float(weights[layer, wire, 0]), wire)
            circuit.rz(float(weights[layer, wire, 1]), wire)
        for wire in range(N_QUBITS - 1):
            circuit.cx(wire, wire + 1)
        circuit.cx(N_QUBITS - 1, 0)
    circuit.measure(0, 0)
    return circuit


def two_qubit_count(circuit: QuantumCircuit) -> int:
    return sum(count for name, count in circuit.count_ops().items()
               if name in ("cx", "cz", "ecr", "cy"))


def counts_to_prediction(counts: dict) -> int:
    """A majority of |0> gives +1 (benign). Ties give -1, so that the rule
    matches the simulator, where an expectation of exactly zero is assigned
    to the attack class."""
    return 1 if counts.get("0", 0) > counts.get("1", 0) else -1


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


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=N_SAMPLES,
                        help="test samples to evaluate per model")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and transpile without submitting")
    args = parser.parse_args()

    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")

    # A stratified prefix of the test set, so the evaluated subset carries
    # the same class balance as the whole.
    n_attack = int(round(args.n * np.mean(y_test == -1)))
    attack_idx = np.where(y_test == -1)[0][:n_attack]
    benign_idx = np.where(y_test == 1)[0][:args.n - n_attack]
    sample_idx = np.sort(np.concatenate([attack_idx, benign_idx]))

    log(f"evaluating {len(sample_idx)} test samples "
        f"({len(attack_idx)} attack, {len(benign_idx)} benign)")

    available = [(name, path) for name, path in MODELS
                 if os.path.exists(path)]
    for name, path in MODELS:
        if not os.path.exists(path):
            log(f"  {path} not found; skipping {name}")
    if not available:
        log("no model files found")
        sys.exit(1)

    parameters = {name: np.load(path) for name, path in available}

    log("\nbuilding circuits")
    tagged_circuits = []
    for name in parameters:
        for index in sample_idx:
            tagged_circuits.append(
                (name, int(index),
                 build_circuit(X_test[index], parameters[name]))
            )
    log(f"  {len(tagged_circuits)} circuits "
        f"({len(sample_idx)} per model, {len(parameters)} models)")

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

    token = os.environ.get("IBM_API_KEY")
    instance = os.environ.get("IBM_INSTANCE")
    if not token or not instance:
        log("\nIBM_API_KEY or IBM_INSTANCE is not set in this terminal.")
        sys.exit(1)

    service = QiskitRuntimeService(channel="ibm_cloud", token=token,
                                   instance=instance)
    backend = service.backend(BACKEND_NAME)
    log(f"\nbackend {backend.name}, {backend.num_qubits} qubits")

    pass_manager = generate_preset_pass_manager(
        backend=backend, optimization_level=OPTIMIZATION_LEVEL)

    log("transpiling")
    transpiled = [pass_manager.run(circuit)
                  for _, _, circuit in tagged_circuits]

    logical, physical = tagged_circuits[0][2], transpiled[0]
    transpilation = {
        "optimisation_level": OPTIMIZATION_LEVEL,
        "logical_depth": int(logical.depth()),
        "logical_two_qubit_gates": int(two_qubit_count(logical)),
        "transpiled_depth": int(physical.depth()),
        "transpiled_two_qubit_gates": int(two_qubit_count(physical)),
    }
    log(f"  depth {transpilation['logical_depth']} -> "
        f"{transpilation['transpiled_depth']}")
    log(f"  two-qubit gates {transpilation['logical_two_qubit_gates']} -> "
        f"{transpilation['transpiled_two_qubit_gates']}")

    if args.dry_run:
        log("\ndry run; nothing submitted")
        return

    log(f"\nsubmitting one job with {len(transpiled)} circuits "
        f"at {N_SHOTS} shots")
    started = time.time()
    job = Sampler(backend).run(transpiled, shots=N_SHOTS)
    log(f"  job id {job.job_id()}")
    log("  monitor at https://quantum.cloud.ibm.com/jobs")
    result = job.result()
    log(f"  returned after {time.time() - started:.0f}s")

    predictions: dict[str, dict[int, int]] = {}
    for position, (name, index, _) in enumerate(tagged_circuits):
        counts = result[position].data.c.get_counts()
        predictions.setdefault(name, {})[index] = counts_to_prediction(counts)

    output = {
        "backend": BACKEND_NAME,
        "job_id": job.job_id(),
        "n_shots": N_SHOTS,
        "n_samples": int(len(sample_idx)),
        "n_attack": int(len(attack_idx)),
        "n_benign": int(len(benign_idx)),
        "transpilation": transpilation,
        "models": [],
    }

    log(f"\n{'model':>16} {'simulator':>11} {'hardware':>11} "
        f"{'correct':>9} {'95% CI':>16}")
    log("-" * 68)

    truth = y_test[sample_idx]
    for name, params in parameters.items():
        simulated = simulator_predict(X_test[sample_idx], params)
        simulator_accuracy = float(np.mean(simulated == truth))

        hardware = np.array([predictions[name][int(i)] for i in sample_idx])
        correct = int(np.sum(hardware == truth))
        hardware_accuracy = correct / len(sample_idx)
        low, high = wilson_interval(correct, len(sample_idx))

        # Fraction of samples where hardware and simulator return the same
        # label, which separates device error from model accuracy.
        agreement = float(np.mean(hardware == simulated))

        log(f"{name:>16} {simulator_accuracy:>10.1%} "
            f"{hardware_accuracy:>10.1%} {correct:>5}/{len(sample_idx):<3} "
            f"[{low:>5.2f},{high:>5.2f}]")

        output["models"].append({
            "name": name,
            "simulator_accuracy": round(simulator_accuracy, 4),
            "hardware_accuracy": round(hardware_accuracy, 4),
            "n_correct": correct,
            "n_evaluated": int(len(sample_idx)),
            "hardware_accuracy_ci95": [round(low, 4), round(high, 4)],
            "simulator_hardware_agreement": round(agreement, 4),
            "per_sample": {
                str(int(i)): {
                    "true": int(y_test[i]),
                    "simulator": int(simulated[k]),
                    "hardware": int(predictions[name][int(i)]),
                }
                for k, i in enumerate(sample_idx)
            },
        })

    os.makedirs("results", exist_ok=True)
    with open("results/hardware_clean_vs_poisoned.json", "w",
              encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    log("\nsaved results/hardware_clean_vs_poisoned.json")


if __name__ == "__main__":
    main()
