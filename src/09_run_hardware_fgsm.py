"""FGSM transfer to IBM Fez, measured against the wide-margin model.

Adversarial examples are generated on the simulator, then the original
and perturbed inputs are classified on hardware. Comparing simulator and
hardware attack success at the same budget is what measures transfer.

All circuits go in one job, so every circuit shares one calibration
state. Transpiled depth and two-qubit gate count are recorded alongside
the results, which is what allows transpilation overhead to be reported
separately from device noise.

The two budgets are far enough apart in simulation that device noise
cannot blur them together.

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
models/noisy_0.0_params.npy

Output
------
results/hardware_fgsm_adam.json

Usage
-----
    python -u 09_run_hardware_fgsm.py
    python -u 09_run_hardware_fgsm.py --dry-run
    python -u 09_run_hardware_fgsm.py --eps 0.30
"""

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
OPT_LEVEL = 1
EPSILONS = [0.30, 0.50]
PARAMS_PATH = "models/noisy_0.0_params.npy"


def log(*args):
    print(*args, flush=True)


# ----------------------------------------------------------------------------
# Simulator side: generate the adversarial examples
# ----------------------------------------------------------------------------

dev_sim = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev_sim, interface="autograd", diff_method="backprop")
def vqc_sim(x, params):
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


def sim_predict(X, params):
    X = pnp.array(np.atleast_2d(X), requires_grad=False)
    p = pnp.array(params, requires_grad=False)
    raw = np.array(pnp.reshape(vqc_sim(X, p), (-1,)), dtype=float)
    return np.where(raw > 0, 1, -1)


def fgsm(x, y_true, params, epsilon):
    xi = pnp.array(np.array(x, dtype=float), requires_grad=True)
    p = pnp.array(params, requires_grad=False)
    grad = qml.grad(lambda inp: -float(y_true) * vqc_sim(inp, p))(xi)
    return np.clip(np.array(x) + epsilon * np.sign(np.array(grad)), 0, np.pi)


# ----------------------------------------------------------------------------
# Hardware side
# ----------------------------------------------------------------------------


def build_circuit(x, params):
    w = np.asarray(params).reshape(N_LAYERS, N_QUBITS, 2)
    qc = QuantumCircuit(N_QUBITS, 1)
    for i in range(N_QUBITS):
        qc.ry(float(x[i]), i)
    for layer in range(N_LAYERS):
        for i in range(N_QUBITS):
            qc.ry(float(w[layer, i, 0]), i)
            qc.rz(float(w[layer, i, 1]), i)
        for i in range(N_QUBITS - 1):
            qc.cx(i, i + 1)
        qc.cx(N_QUBITS - 1, 0)
    qc.measure(0, 0)
    return qc


def two_qubit_count(qc):
    return sum(n for name, n in qc.count_ops().items()
               if name in ("cx", "cz", "ecr", "cy", "czz"))


def counts_to_pred(counts):
    """|0> majority means +1 (benign). Ties go to -1, matching the simulator,
    where an expectation of exactly zero is treated as the attack class."""
    c0 = counts.get("0", 0)
    c1 = counts.get("1", 0)
    return 1 if c0 > c1 else -1


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(center - half), float(center + half)


# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="build and transpile everything, submit nothing")
    parser.add_argument("--eps", type=float, nargs="*", default=None)
    parser.add_argument("--n", type=int, default=N_SAMPLES)
    args = parser.parse_args()

    epsilons = args.eps if args.eps else EPSILONS

    params = np.load(PARAMS_PATH)
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")

    # Only attack samples the simulator classifies correctly. Attacking a
    # sample the model already gets wrong is not an attack success.
    attack_idx = np.where(y_test == -1)[0][:args.n]
    preds = sim_predict(X_test[attack_idx], params)
    keep = attack_idx[preds == -1]
    log(f"model: {PARAMS_PATH}")
    log(f"attack samples: {len(keep)} correctly classified of {len(attack_idx)}")
    log(f"epsilons: {epsilons}")

    # Build every circuit up front: originals once, adversarial per epsilon.
    log("\nbuilding circuits")
    jobs = []           # (tag, index, QuantumCircuit)
    for i in keep:
        jobs.append(("orig", int(i), build_circuit(X_test[i], params)))
    for eps in epsilons:
        for i in keep:
            x_adv = fgsm(X_test[i], y_test[i], params, eps)
            jobs.append((f"adv_{eps}", int(i), build_circuit(x_adv, params)))
    log(f"  {len(jobs)} circuits total "
        f"({len(keep)} original + {len(keep)} per epsilon)")

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

    token = os.environ.get("IBM_API_KEY")
    instance = os.environ.get("IBM_INSTANCE")
    if not token or not instance:
        log("\nIBM_API_KEY or IBM_INSTANCE not set in this terminal. See the")
        log("docstring at the top of this file.")
        sys.exit(1)

    service = QiskitRuntimeService(channel="ibm_cloud", token=token,
                                   instance=instance)
    backend = service.backend(BACKEND_NAME)
    log(f"\nbackend {backend.name}, {backend.num_qubits} qubits")

    pm = generate_preset_pass_manager(backend=backend,
                                      optimization_level=OPT_LEVEL)

    log("transpiling")
    isa = [pm.run(qc) for _, _, qc in jobs]

    logical = jobs[0][2]
    physical = isa[0]
    transpilation = {
        "optimization_level": OPT_LEVEL,
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
        log("\ndry run, nothing submitted")
        return

    log(f"\nsubmitting one job with {len(isa)} circuits at {N_SHOTS} shots")
    t0 = time.time()
    sampler = Sampler(backend)
    job = sampler.run(isa, shots=N_SHOTS)
    log(f"  job id {job.job_id()}")
    log("  waiting, monitor at https://quantum.cloud.ibm.com/jobs")

    result = job.result()
    log(f"  returned after {time.time() - t0:.0f}s")

    preds_by_tag = {}
    for k, (tag, idx, _) in enumerate(jobs):
        counts = result[k].data.c.get_counts()
        preds_by_tag.setdefault(tag, {})[idx] = counts_to_pred(counts)

    orig = preds_by_tag["orig"]
    hw_correct = [i for i in orig if orig[i] == -1]
    log(f"\nhardware classified {len(hw_correct)}/{len(keep)} originals "
        f"correctly as attacks")

    out = {
        "backend": BACKEND_NAME,
        "job_id": job.job_id(),
        "model": PARAMS_PATH,
        "n_shots": N_SHOTS,
        "n_attack_samples": int(len(keep)),
        "transpilation": transpilation,
        "hardware_clean_correct": len(hw_correct),
        "epsilons": [],
    }

    log(f"\n{'eps':>6} {'sim ASR':>9} {'hw ASR':>9} {'fooled':>9} {'95% CI':>16}")
    log("-" * 54)
    for eps in epsilons:
        adv = preds_by_tag[f"adv_{eps}"]
        # Success requires the sample to be an attack the hardware got right,
        # then flipped by the perturbation.
        fooled = sum(1 for i in hw_correct if adv[i] == 1)
        tested = len(hw_correct)
        hw_asr = fooled / tested if tested else 0.0

        X_adv = np.stack([fgsm(X_test[i], y_test[i], params, eps) for i in keep])
        sim_asr = float(np.mean(sim_predict(X_adv, params) == 1))

        lo, hi = wilson(fooled, tested)
        log(f"{eps:>6.2f} {sim_asr:>8.1%} {hw_asr:>8.1%} "
            f"{fooled:>5}/{tested:<3} [{lo:>5.2f},{hi:>5.2f}]")

        out["epsilons"].append({
            "epsilon": eps,
            "simulator_asr": round(sim_asr, 4),
            "hardware_asr": round(hw_asr, 4),
            "n_fooled": fooled,
            "n_tested": tested,
            "hardware_asr_ci95": [round(lo, 4), round(hi, 4)],
            "per_sample": {str(i): int(adv[i]) for i in sorted(adv)},
        })

    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/hardware_fgsm_adam.json", "w"), indent=2)
    log("\nsaved results/hardware_fgsm_adam.json")


if __name__ == "__main__":
    main()
