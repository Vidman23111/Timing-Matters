# Timing Matters: A Backdoor Attack on Audio-based Spiking Neural Networks

This repository contains the code used to train and backdoor a Spiking Neural Network (SNN) on the Spiking Heidelberg Digits (SHD) dataset, as part of a bachelor's thesis investigating backdoor attacks on audio-based SNNs.

## Overview

The pipeline trains an SNN on the SHD dataset, then poisons a copy of the training data by injecting a rectangular trigger pattern into the spike representation of selected samples and relabelling them to a target class. The backdoored model is evaluated on both clean accuracy and Attack Success Rate (ASR), and a set of parameter sweeps and analysis plots are produced.

## Requirements

- Python 3.x
- `torch`
- `snntorch`
- `h5py`
- `numpy`
- `matplotlib`

Install with:

```bash
pip install torch snntorch h5py numpy matplotlib
```

## Dataset

This project requires the **Spiking Heidelberg Digits (SHD)** dataset, which is **not included** in this repository.

Download the dataset (`shd_train.h5` and `shd_test.h5`, or their alternate `_1` / `.hdf5` naming) and place the files inside a `data/` folder next to `backdoor attack.py`:

```
backdoor attack.py
data/
    shd_train.h5
    shd_test.h5
```

The script searches recursively inside `data/` for the expected filenames, so subfolders are fine too.

## Run Controls

At the top of `backdoor attack.py` there are several flags you can edit before each run:

| Flag | Purpose |
|---|---|
| `SKIP_CLEAN_TRAINING` | If `True`, loads the saved clean model checkpoint (`shd_snntorch_best.pt`) instead of retraining from scratch. The clean model only needs to be trained once. |
| `SKIP_BACKDOOR_TRAINING` | If `True`, loads the backdoor checkpoint from the active config's results folder instead of retraining. Useful for re-running evaluation/plots only. |
| `SKIP_SWEEPS` | If `True`, skips all parameter sweeps (duration, channel width, position, poison ratio) and only produces the main training curves and evaluation plots. Sweeps are slow since several retrain a fresh model per value. |
| `DOUBLE_TRIGGER` | If `True`, adds a second trigger offset by 200 channels from the first, fired at the same time window. |
| `ACTIVE_CONFIG` | Selects which attack configuration to run. Must be one of the keys in `ATTACK_CONFIGS` (see below). |

## Attack Configurations

Three named configurations are defined in `ATTACK_CONFIGS`, corresponding to the three main experiments in the thesis:

- **`low_poison`** — a small, subtle trigger with a low poison ratio. Produces a low ASR, illustrating that an undersized trigger combined with insufficient poisoning fails to implant a reliable backdoor.
- **`high_poison`** — a large trigger with a high poison ratio. Produces a very high ASR but at the cost of realism, since the trigger is large enough to effectively corrupt the input rather than act as a subtle pattern.
- **`optimal_setup`** — a compact, balanced trigger placed in a low-activity channel region, achieving a high ASR with minimal impact on clean accuracy and a small visual/statistical footprint.

Each configuration defines: `target_label`, `poison_ratio`, `trigger_start`, `trigger_duration`, `channel_start`, `channel_end`, `trigger_value`, and `seed`.

Switch between them by setting `ACTIVE_CONFIG` to the desired key.

## Output

Each configuration saves its checkpoints and plots into its own subfolder (e.g. `optimal setup/`, `low poison/`, `high poison/`), named according to `_FOLDER_NAMES` in the script. The clean model checkpoint and its training curve plots are saved in the script's root directory instead, since the clean model is shared across all configurations.

**Note:** output paths are currently hardcoded relative to the script's own directory (`SCRIPT_DIR`) rather than configurable via a command-line argument or environment variable. If you move the script, the `data/` folder and results folders are expected to sit alongside it.

## Inspecting Results

`inspect results.py` is a small helper script that loads the saved checkpoints from a set of result folders and prints a summary table of best/final epoch, clean accuracy, and ASR, along with the backdoor configuration used for each run. Edit the `folders` list at the top of the script to point at the result folders you want to inspect.
