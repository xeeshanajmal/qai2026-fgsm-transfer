"""Convert the narrow-margin hardware transfer run into the standard format.

The FGSM transfer experiment was run against both classifiers. The wide-margin
run was written by run_hardware_fgsm.py in the current result format. The
narrow-margin run predates that script and stored its output under a different
schema, without Wilson intervals.

This reads the older file, recomputes the intervals from the recorded counts,
and writes it in the same shape as the wide-margin result, so that
make_figures.py reads both through one code path and neither is entered by
hand.

The transfer panel of Figure 1 uses both files. Without this conversion it
falls back to the wide-margin results alone, which halves the evidence behind
the transfer claim.

Input
-----
results/transferability_hw_50_expA.json

Output
------
results/hardware_fgsm_cobyla.json

Usage
-----
    python -u convert_narrow_margin_hardware.py
    python -u convert_narrow_margin_hardware.py --input path/to/file.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

DEFAULT_INPUT = "results/transferability_hw_50_expA.json"
OUTPUT = "results/hardware_fgsm_cobyla.json"


def log(message: str) -> None:
    print(message, flush=True)


def wilson_interval(successes: int, trials: int,
                    z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, matching the one used by the other scripts."""
    if trials == 0:
        return 0.0, 0.0
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = z * np.sqrt(p * (1 - p) / trials
                         + z * z / (4 * trials * trials)) / denominator
    return float(centre - spread), float(centre + spread)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        log(f"{args.input} not found. Figure 1 will use the wide-margin "
            f"results alone.")
        return

    with open(args.input, encoding="utf-8") as handle:
        source = json.load(handle)

    simulator = source.get("simulator_reference", {}).get("fgsm", [])
    rows = source.get("fgsm_hardware", [])
    if len(simulator) != len(rows):
        log("simulator reference and hardware rows differ in length; "
            "cannot pair them reliably")
        return

    converted = []
    log(f"{'eps':>6} {'simulator':>11} {'hardware':>11} {'95% CI':>16}")
    log("-" * 48)

    for row, simulated in zip(rows, simulator):
        fooled = int(row["n_fooled"])
        tested = int(row["n_correctly_classified"])
        low, high = wilson_interval(fooled, tested)

        log(f"{row['epsilon']:>6.2f} {simulated:>10.1%} "
            f"{row['hw_asr']:>10.1%} [{low:>5.2f},{high:>5.2f}]")

        converted.append({
            "epsilon": float(row["epsilon"]),
            "simulator_asr": round(float(simulated), 4),
            "hardware_asr": round(float(row["hw_asr"]), 4),
            "n_fooled": fooled,
            "n_tested": tested,
            "hardware_asr_ci95": [round(low, 4), round(high, 4)],
        })

    output = {
        "backend": source.get("backend", "ibm_fez"),
        "model": "narrow_margin",
        "n_shots": source.get("n_shots"),
        "n_attack_samples": source.get("n_samples"),
        "epsilons": converted,
        "note": ("Converted from the original transfer run. Intervals "
                 "recomputed from the recorded counts."),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    log(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
