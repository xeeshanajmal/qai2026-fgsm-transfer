# Adversarial robustness of a variational quantum classifier for network intrusion detection

Code, trained parameters and results for the paper *FGSM Attack
Transferability from Simulation to Quantum Hardware in Network Intrusion
Detection*, IEEE International Conference on Quantum Artificial
Intelligence (QAI) 2026.

The numbers in the paper come from the JSON files in `results/`. The
scripts here made those files.

## Files

```
src/        experiment scripts, numbered in run order
data/       preprocessed splits (the raw CICIDS2017 CSVs are not included)
models/     trained circuit parameters
results/    one JSON per experiment
figures/    the two figures in the paper
```

## Requirements

```
python >= 3.10
pennylane == 0.44.1
qiskit-ibm-runtime
numpy, scipy, scikit-learn, pandas, matplotlib
```

Install with `pip install -r requirements.txt`.

Scripts 08 and 09 run on real hardware, so they need an IBM Quantum
account. Copy `.env.example` and put your own token and instance in it.
Then set them as environment variables. The scripts read credentials from
the environment only.

## Data

The raw capture is not here. Download CICIDS2017 from the
[Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
and put `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` in `data/`.
That is the only file used.

`01_preprocess.py` builds the four `.npy` splits. You should get 1134
attack and 866 benign samples in training. In test, 284 attack and 216
benign. Different counts mean a different capture file, or a different
scikit-learn version.

## How to run

Run the scripts in number order. Scripts 04 to 07 need 01, 02 and 03 to
have run first.

| Script | Produces | Runtime |
| --- | --- | --- |
| `01_preprocess.py` | the four splits in `data/` | seconds |
| `02_train_narrow_margin.py` | `models/clean_params.npy` | seconds |
| `03_retrain_poisoning_and_noise.py` | poisoned and noisy models, two result files | several hours |
| `04_epsilon_sweep.py` | `results/epsilon_sweep.json` | under a minute |
| `05_compare_attack_guidance.py` | `results/gradient_vs_untargeted.json` | under a minute |
| `06_finish_rq3.py` | noiseless baseline and the noise FGSM sweep | about an hour |
| `07_margin_control.py` | `results/margin_control.json` | a few minutes |
| `08_run_hardware_models.py` | `results/hardware_clean_vs_poisoned.json` | queue time |
| `09_run_hardware_fgsm.py` | `results/hardware_fgsm_adam.json` | queue time |
| `10_convert_hardware_results.py` | `results/hardware_fgsm_cobyla.json` | seconds |
| `make_figures.py` | the two figures | seconds |

Script 03 is the slow one. It simulates density matrices, which costs
about sixteen minutes per initialization. It trains three initializations
at each of three noise levels.

The hardware scripts send all their circuits in one job. Every circuit in
a job then shares the same calibration state. Both scripts take
`--dry-run`, which transpiles everything and prints the circuit statistics
without sending anything.

## Which file makes which result

| Paper | Result file | Script |
| --- | --- | --- |
| Table I, poisoning | `poisoning_retrained.json` | 03 |
| Table II, FGSM against untargeted | `gradient_vs_untargeted.json` | 05 |
| Table III, simulation against hardware | `hardware_clean_vs_poisoned.json`, `hardware_fgsm_adam.json` | 08, 09 |
| Table IV, depolarizing noise | `noise_retrained.json`, `noise_fgsm_sweep.json`, `noiseless_400_baseline.json` | 03, 06 |
| Figure 1, transfer to hardware | `hardware_fgsm_adam.json`, `hardware_fgsm_cobyla.json` | 09, 10 |
| Figure 2, accuracy and margin under noise | same as Table IV | 03, 06 |
| Section IV-A, clean baseline | `clean_history.json`, `noise_retrained.json` | 02, 03 |
| Section IV-C, decision margins | `epsilon_sweep.json` | 04 |
| Section VI, margin control | `margin_control.json` | 07 |

## Trained models

| File | What it is |
| --- | --- |
| `clean_params.npy` | narrow-margin classifier, COBYLA, 86.8% accuracy at margin 0.07 |
| `noisy_0.0_params.npy` | wide-margin classifier, Adam, 92.2% accuracy at margin 0.54 |
| `noisy_0.0_params_400.npy` | noiseless model trained on the 400-sample subset, to compare across noise levels |
| `noisy_0.001_params.npy`, `noisy_0.01_params.npy`, `noisy_0.05_params.npy` | one per depolarizing noise level |
| `poisoned_5pct_params.npy`, `poisoned_10pct_params.npy`, `poisoned_15pct_params.npy` | one per label-flipping rate |

## Hardware runs

All runs used IBM Fez, a 156-qubit Heron r2 processor. 1024 shots per
circuit, through Qiskit Runtime SamplerV2. Transpilation was at
optimization level 1.

Every run submitted the same logical circuit: depth 20 with 12 two-qubit
gates. Transpilation maps it onto physical qubits, which changes both
numbers on each submission.

| Result | Script | Circuits | Transpiled depth | Two-qubit gates |
| --- | --- | --- | --- | --- |
| Clean and poisoned model accuracy | 08 | 100 | 100 | 27 |
| FGSM transfer, wide-margin model | 09 | 150 | 87 | 42 |
| FGSM transfer, narrow-margin model | earlier code, superseded by 09 | 400 | not recorded | not recorded |

The narrow-margin transfer run predates the scripts in this repository.
Its output is `results/transferability_hw_50_expA.json`, which script 10
converts into the format the figures read. The run used the same circuit
and the same optimization level as script 09.

The two recorded depths differ. The transpiler picks physical qubits by
heuristic, so the same circuit maps differently on each submission. Both
runs still agreed with simulation.

## What you cannot reproduce

Hardware results depend on the calibration state of the device that day.
Send the same circuits again and you get different counts. The JSON files
are the record of what the device did.

`models/clean_params.npy` cannot be regenerated. Its initial parameters
came from the global NumPy random state instead of a seeded generator.
`02_train_narrow_margin.py` now seeds its initialization, so new runs are
reproducible among themselves. They land somewhere else, though, and the
spread is wide. A gradient-free optimizer on a stochastic objective can
collapse to predicting the majority class. The script says so when that
happens.

The rest is seeded. Accuracy still varies across initializations, so the
scripts report mean and standard deviation instead of one run. Section VI
of the paper explains why.

## Citation

```bibtex
@inproceedings{ajmal2026fgsm,
  author    = {Ajmal, Zeeshan and Halunen, Kimmo and Mau{\ss}ner, Marc
               and Reers, Volker and Khan, Arif Ali},
  title     = {{FGSM} Attack Transferability from Simulation to Quantum
               Hardware in Network Intrusion Detection},
  booktitle = {IEEE International Conference on Quantum Artificial
               Intelligence (QAI)},
  year      = {2026},
}
```

## License

The code is MIT. See `LICENSE`.

CICIDS2017 comes from the Canadian Institute for Cybersecurity under its
own terms. It is not redistributed here.

## Acknowledgment

This work was supported by the Business Finland project (24304955)
SeQuSoS. We acknowledge the use of IBM Quantum services. The views
expressed are those of the authors and do not reflect the official policy
or position of IBM or the IBM Quantum team.
