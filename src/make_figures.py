"""Generate the two figures in the paper from the stored results.

Both are derived from the JSON files written by the experiment scripts,
so they cannot drift from the numbers reported in the tables.

Figure 1, transfer_scatter
    Attack success measured on IBM Fez against the value predicted in
    simulation, for both classifiers across six perturbation budgets.
    Points on the identity line transferred without loss. This is the
    direct form of the transfer claim: two aggregate bars would show the
    same conclusion with less evidence behind it.

Figure 2, noise_bar
    Clean accuracy and decision margin under depolarizing noise, each
    relative to its noiseless value so both fit one axis. The divergence
    between them is the point, and a twin axis in absolute units would
    make the two harder to compare.

Inputs
------
results/hardware_fgsm_adam.json     wide-margin, two budgets
results/hardware_fgsm_cobyla.json   narrow-margin, four budgets
results/noise_retrained.json        clean accuracy per noise level
results/noiseless_400_baseline.json noiseless model at matched size
results/noise_fgsm_sweep.json       decision margin per noise level

Outputs
-------
figures/transfer_scatter.{pdf,png}
figures/noise_bar.{pdf,png}
figures/captions.txt

Usage
-----
    python -u make_figures.py
    python -u make_figures.py --png-only
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Okabe-Ito palette: distinguishable under common colour vision
# deficiencies and in greyscale.
COLOR_NARROW = "#D55E00"
COLOR_WIDE = "#0072B2"
COLOR_MARGIN = "#666666"
COLOR_REFERENCE = "#999999"

# Source widths are kept close to the printed column width so the figures
# are scaled down only slightly. A wider source shrinks the tick labels
# relative to the rest of the page.
SIZE_TRANSFER = (3.9, 3.5)
SIZE_NOISE = (4.4, 3.2)

PLOT_STYLE = {
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
}

CAPTIONS = {
    "transfer_scatter": (
        "Attack success on IBM Fez against the value predicted in "
        "simulation, six budgets across both classifiers. The dotted line "
        "marks equality. At the leftmost point the perturbation is "
        "negligible, so the measured successes are device error."
    ),
    "noise_bar": (
        "Clean accuracy and decision margin under depolarizing noise, "
        "relative to their noiseless values."
    ),
}


def log(message: str) -> None:
    print(message, flush=True)


def load_results(results_dir: str, name: str,
                 required: bool = True) -> Any | None:
    path = os.path.join(results_dir, name)
    if not os.path.exists(path):
        if required:
            log(f"  {path} not found; skipping figures that need it")
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def percent(value: float) -> float:
    return 100.0 * value


def asymmetric_error(values: Sequence[float],
                     intervals: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert [lower, upper] bounds into the 2xN array matplotlib wants."""
    lower = [max(0.0, v - ci[0]) for v, ci in zip(values, intervals)]
    upper = [max(0.0, ci[1] - v) for v, ci in zip(values, intervals)]
    return np.array([lower, upper])


def save(figure: plt.Figure, stem: str, fig_dir: str,
         png_only: bool) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    figure.savefig(os.path.join(fig_dir, f"{stem}.png"))
    if not png_only:
        figure.savefig(os.path.join(fig_dir, f"{stem}.pdf"))
    plt.close(figure)
    log(f"  {stem}")


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------


def figure_transfer(results_dir: str, fig_dir: str, png_only: bool) -> None:
    wide = load_results(results_dir, "hardware_fgsm_adam.json")
    narrow = load_results(results_dir, "hardware_fgsm_cobyla.json",
                          required=False)
    if not wide:
        return

    sources = []
    if narrow:
        sources.append((narrow, COLOR_NARROW, "o", "Narrow margin"))
    sources.append((wide, COLOR_WIDE, "s", "Wide margin"))

    figure, ax = plt.subplots(figsize=SIZE_TRANSFER)
    ax.plot([0, 100], [0, 100], color=COLOR_MARGIN, linewidth=1,
            linestyle=":", zorder=1)

    for source, colour, marker, label in sources:
        rows = source["epsilons"]
        measured = [percent(row["hardware_asr"]) for row in rows]
        intervals = [[percent(bound) for bound in row["hardware_asr_ci95"]]
                     for row in rows]
        ax.errorbar([percent(row["simulator_asr"]) for row in rows],
                    measured,
                    yerr=asymmetric_error(measured, intervals),
                    fmt=marker, color=colour, markersize=5, capsize=3,
                    linewidth=1.2, linestyle="none", label=label, zorder=3)
        log(f"    {label}: {len(rows)} paired measurements")

    ax.set_xlabel("Simulator attack success (%)")
    ax.set_ylabel("IBM Fez attack success (%)")
    ax.set_xlim(-4, 100)
    ax.set_ylim(-4, 100)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)

    figure.tight_layout()
    save(figure, "transfer_scatter", fig_dir, png_only)


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------


def figure_noise(results_dir: str, fig_dir: str, png_only: bool) -> None:
    noise = load_results(results_dir, "noise_retrained.json")
    attacks = load_results(results_dir, "noise_fgsm_sweep.json")
    baseline = load_results(results_dir, "noiseless_400_baseline.json",
                            required=False)
    if not noise or not attacks:
        return

    accuracy_by_level = {row["noise_prob"]: row for row in noise}
    levels = [row["noise_prob"] for row in attacks]
    labels = [row["label"] for row in attacks]

    accuracy = []
    for level in levels:
        # The noiseless point comes from the matched-size baseline so that
        # every level reflects the same amount of training data.
        if level == 0.0 and baseline:
            accuracy.append(percent(baseline["clean"]["test_accuracy_mean"]))
        else:
            accuracy.append(percent(
                accuracy_by_level[level]["clean_accuracy_mean"]))

    margin = [row["margin_median"] for row in attacks]
    relative_accuracy = [100.0 * v / accuracy[0] for v in accuracy]
    relative_margin = [100.0 * v / margin[0] for v in margin]

    for index, label in enumerate(labels):
        log(f"    {label:>10}: accuracy {relative_accuracy[index]:.1f}%, "
            f"margin {relative_margin[index]:.1f}% of noiseless")

    positions = np.arange(len(labels))
    width = 0.38

    figure, ax = plt.subplots(figsize=SIZE_NOISE)
    ax.axhline(100, color=COLOR_REFERENCE, linewidth=0.8, linestyle=":",
               zorder=2)
    ax.bar(positions - width / 2, relative_accuracy, width,
           color=COLOR_WIDE, label="Clean accuracy", zorder=3)
    ax.bar(positions + width / 2, relative_margin, width,
           color=COLOR_MARGIN, label="Decision margin", zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Depolarizing probability")
    ax.set_ylabel("Relative to noiseless (%)")
    ax.set_ylim(0, 108)
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    # Every x position carries a full-height accuracy bar, so no location
    # inside the axes is clear. The legend goes below the axis instead.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              frameon=False)

    figure.tight_layout()
    save(figure, "noise_bar", fig_dir, png_only)


# ---------------------------------------------------------------------------


def write_captions(fig_dir: str) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "captions.txt")
    with open(path, "w", encoding="utf-8") as handle:
        for stem, text in CAPTIONS.items():
            handle.write(f"{stem}\n{text}\n\n")
    log("  captions.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--fig-dir", default="figures")
    parser.add_argument("--png-only", action="store_true",
                        help="skip PDF output while iterating")
    args = parser.parse_args()

    plt.rcParams.update(PLOT_STYLE)
    log(f"reading {args.results_dir}/, writing {args.fig_dir}/")
    figure_transfer(args.results_dir, args.fig_dir, args.png_only)
    figure_noise(args.results_dir, args.fig_dir, args.png_only)
    write_captions(args.fig_dir)


if __name__ == "__main__":
    main()
