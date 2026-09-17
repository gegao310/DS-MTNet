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
├── feature_schema.py         # Unified 14-D cycle-level feature schema (paper Table 1)
├── XJTU_processing.py        # XJTU dataset: cycle-level feature extraction -> .pkl
├── NASA_dataprocessing.py    # NASA dataset: cycle-level feature extraction -> .pkl
├── XJTU_experiment.py        # Main experiments (Exp1-Exp4): benchmark, ablation, transfer
├── NASA_experiment.py        # NASA experiments: LOO within-group + cross-group transfer
├── battery_alert.py          # Hierarchical early warning analysis (Algorithm 1)
└── requirements.txt
```

## Unified 14-D Feature Vector

Both datasets are mapped onto a **single 14-dimensional cycle-level feature vector**
(`F = 14`), so that network weights can be transferred across datasets. The schema is
defined once in `feature_schema.py` and imported by both extraction scripts; column
names, order and units match **Table 1** of the paper.

| Phase | Feature | Symbol | Unit | XJTU | NASA |
|---|---|---|---|---|---|
| CC charge | Constant-current duration | `t_CC` | min | measured | zero-padded |
| CC charge | Constant-current input energy | `E_CC` | Wh | measured | zero-padded |
| CC charge | Rate of voltage change | `dVdt` | V/min | measured | zero-padded |
| CC charge | Shannon entropy of voltage | `H_V_CC` | bits | measured | zero-padded |
| CV charge | Constant-voltage input capacity | `Q_CV` | Ah | measured | zero-padded |
| CV charge | Constant-voltage input energy | `E_CV` | Wh | measured | zero-padded |
| CV charge | Shannon entropy of current | `H_I_CV` | bits | measured | zero-padded |
| Discharge | Discharge capacity | `Q_dis` | Ah | measured | measured |
| Discharge | Discharge energy | `E_dis` | Wh | measured | measured |
| Discharge | Mean discharge current | `I_dis_mean` | A | measured | measured |
| Discharge | Mean discharge voltage | `V_dis_mean` | V | measured | measured |
| Discharge | Minimum discharge voltage | `V_dis_min` | V | measured | measured |
| Discharge | Std. of discharge voltage | `V_dis_std` | V | measured | measured |
| Discharge | Discharge voltage decay slope | `k_V_dis` | V/step | measured | measured |

Energy / capacity are obtained by numerical integration; integration time is converted
to hours, so that `int(V*I dt)` is directly in Wh and `int(I dt)` in Ah. Because the
NASA dataset contains only discharge cycles, its seven charge-phase channels are filled
with zeros. If you edit one extraction script, keep the other in sync via
`feature_schema.py` — a mismatch raises a `ValueError` at run time.

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

This produces `unified_Batch-{1..4}.pkl`. Each file contains `battery_id`,
`cycle_count`, the **14-D** unified feature vector described above, and the
`Target_T_max` / `Target_T_mean` / `Target_T_rise` / `Target_SOH` labels.

### NASA

Edit `RAW_DATA_DIR` / `OUT_FEATURE_DIR` in `NASA_dataprocessing.py` (or call
`preprocess_all(raw_dir, out_dir)`), then run:

```bash
python NASA_dataprocessing.py
```

This produces `B0005_features.pkl`, ..., and a `battery_overview.csv`. Each pickle
carries the same **14-D** unified feature vector (7 measured discharge channels +
7 zero-padded charge channels), so it can be fed to the XJTU-trained model directly.

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
