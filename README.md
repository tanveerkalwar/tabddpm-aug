# TabDDPM-Aug: Adaptive Diffusion-Based Hybrid Augmentation for Imbalanced Tabular Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This is the official code for our paper **"TabDDPM-Aug: Adaptive Diffusion-Based Hybrid Augmentation for Imbalanced Tabular Classification"**, currently submitted to *Expert Systems with Applications*.

## Key Features

- **Adaptive Strategy Selection**: Automatically switches between DBHA (low-density) and FDHA (high-density) regimes.
- **Hybrid Dual-Stream Architecture**: Combines SMOTE-based linear interpolation with an ensemble of K TabDDPM models for robust minority synthesis.
- **Adaptive Quality Filtering**: Employs IQR-based Distance-to-Closest-Real (DCR) metrics to mitigate distributional collapse and prevent memorization.
- **Privacy-Preserving Synthesis**: Significantly reduces Membership Inference Attack (MIA) risk while maintaining competitive classification utility.

## Environment

Tested on (Python 3.12.12, PyTorch 2.9.0). Create a virtualenv or conda environment before installation.

## File Structure

- `TabDDPM_Aug/` – core implementation (config, data loading, generators, evaluation, experiments)
- `scripts/` – experiment entry points (`run_full_experiment.py`)
- `data/` – input CSV files (`adult.csv`, `pima.csv`, `creditcard.csv`)
- `requirements.txt` – Python dependencies
- `setup.py` – package installation script
- `README.md`, `LICENSE.txt` – documentation and license


## Installation

### 1. Clone Repository
```bash
git clone https://github.com/tanveerkalwar/tabddpm-aug.git
cd tabddpm-aug
```

### 2. Install Core Package
```bash
pip install -e .
```

### 3. Verify Installation
```bash
python -c "from TabDDPM_Aug import load_dataset; print('Installation successful')"
```

## Quick Start Example
```bash
# Run
python scripts/run_full_experiment.py --dataset pima

```

## Datasets

This release includes configurations for 13 benchmark datasets used in our paper:

**Note**: The framework can be extended to other tabular datasets by:
1. Adding dataset configuration to `config.py`
2. Placing your CSV in `data/` folder with a `target` column
3. Running experiments with `--dataset your_dataset_name`

For dataset details, file locations, and how to add new datasets, see `data/README.md`


## Troubleshooting


**"CUDA out of memory"**
- Reduce `batch_size` in `config.py`
- Switch to CPU: `device='cpu'`


**Import errors**
```bash
# Ensure proper installation
cd tabddpm-aug
pip install -e .
```

## License

MIT License - see [LICENSE](LICENSE) file.
