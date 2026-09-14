# DS-MTNet: Dual-Stream Multi-Task Network for Joint Thermal–Health Prediction and Early Warning of Lithium-Ion Batteries

This repository provides the official PyTorch implementation of **DS-MTNet**, a dual-stream multi-task network for joint prediction of battery thermal states (maximum / mean temperature) and State of Health (SOH), together with a proactive early warning system. It accompanies the manuscript:

> **Cross-Domain Thermal-Health Prediction and Early Warning for Lithium-Ion Batteries via Dual-Stream Network**

## Overview

DS-MTNet combines two parallel temporal streams:

- A **Temporal Convolutional Network (TCN)** stream that captures short-term, cycle-level thermal dynamics.
- A **Mamba-based selective state-space model** that models long-term degradation (SOH) evolution.

The two streams are fused by an adaptive gated fusion module and followed by task-specific prediction heads for `T_max`, `T_mean`, `SOH`, and an auxiliary temperature-difference task `ΔT`. A high-temperature weighting (HTW) loss focuses training on safety-critical regions, and an MMD-based distribution-alignment strategy (few-shot domain adaptation) supports cross-domain transfer. Model predictions are finally converted into a hierarchical early warning decision (Algorithm 1 in the paper).

## Repository Structure

```
.
├── XJTU_processing.py        # XJTU dataset: cycle-level feature extraction -> .pkl
├── NASA_dataprocessing.py    # NASA dataset: cycle-level feature extraction -> .pkl
├── XJTU_experiment.py        # Main experiments (Exp1-Exp4): benchmark, ablation, transfer
├── NASA_experiment.py        # NASA experiments: LOO within-group + cross-group transfer
├── battery_alert.py          # Hierarchical early warning analysis (Algorithm 1)
└── requirements.txt
```

## Environment

We recommend Python 3.9+ with CUDA-enabled PyTorch. Install dependencies with:

```bash
pip install -r requirements.txt
```

`mamba-ssm` requires a compatible CUDA toolkit; please follow the official
[`mamba-ssm`](https://github.com/state-spaces/mamba) installation guide if the wheel
does not install directly.

## Data Preparation

The two public datasets used in this work are:

- **XJTU** lithium-ion battery dataset (18650 cells, batches 1–4).
- **NASA** Ames Prognostics Center of Excellence battery dataset.

Place the raw datasets in a local directory and run the feature-extraction scripts to
generate cycle-level feature tables. The scripts default to `./XJTU_dataset` and
`./NASA_dataset`; override the paths with command-line arguments or by editing the
configuration at the top of each script.

### XJTU

```bash
python XJTU_processing.py --base_path /path/to/XJTU_dataset --output_dir ./extracted_features
```

This produces `unified_Batch-{1..4}.pkl`, each containing cycle-level features and the
`Target_T_max` / `Target_T_mean` / `Target_SOH` labels.

### NASA

Edit `RAW_DATA_DIR` / `OUT_FEATURE_DIR` in `NASA_dataprocessing.py` (or call
`preprocess_all(raw_dir, out_dir)`), then run:

```bash
python NASA_dataprocessing.py
```

This produces `B0005_features.pkl`, ..., and a `battery_overview.csv`.

## Running Experiments

### XJTU (Exp1–Exp4: benchmark, ablation, cross-condition transfer)

```bash
python XJTU_experiment.py
```

Results are written to `./Results_Main_v4/`, including per-experiment metric CSVs,
prediction CSVs, and figures.

### NASA (leave-one-out within-group + cross-group transfer)

```bash
python NASA_experiment.py
```

Results are written to `./Results_NASA/`.

### Early Warning Analysis

```bash
python battery_alert.py
```

Run this after the experiments; it reads the `predictions/*.csv` outputs and applies the
hierarchical warning logic (baseline establishment, absolute/strong/weak triggers) to
report warning timeliness and lead time. Edit `XJTU_PRED_DIR` / `NASA_PRED_DIR` to point
to your prediction directories.

## Key Hyperparameters

| Component | Setting |
|---|---|
| Historical window length `L` | 40 (XJTU) / 20 (NASA) |
| TCN stream | 2 layers, kernel 3, dilation 1–2, dropout 0.05 |
| Mamba stream | `d_state=128`, `d_conv=4`, `expand=2` |
| SE reduction ratio | 8 (see `SEBlock`) |
| High-temperature thresholds / weights | 35 °C → 1.5, 40 °C → 2.5 |
| MMD coefficient `α` | 0.1 |

## Citation

If you find this repository useful, please cite the corresponding paper.

## License

This code is released for research and reproduction purposes.
