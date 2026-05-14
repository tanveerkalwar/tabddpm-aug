# TabDDPM-Aug: Adaptive Diffusion-Based Hybrid Augmentation for Imbalanced Tabular Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This is the official code for our paper **"TabDDPM-Aug: Adaptive Diffusion-Based Hybrid Augmentation for Imbalanced Tabular Classification"**, currently submitted to *Expert Systems with Applications*.

## Key Features

- **Adaptive Strategy Selection**: Automatically switches between DBHA (low-density) and FDHA (high-density) regimes.
- **Hybrid Dual-Stream Architecture**: Combines SMOTE-based linear interpolation with an ensemble of K TabDDPM models for robust minority synthesis.
- **Adaptive Quality Filtering**: Employs IQR-based Distance-to-Closest-Real (DCR) metrics to mitigate distributional collapse and prevent memorization.
- **Privacy-Preserving Synthesis**: Significantly reduces Membership Inference Attack (MIA) risk while maintaining competitive classification utility.
- **Comprehensive Benchmarking**: Validated against 7 baselines across 3 diverse domains using 9 metrics for utility, fidelity, and privacy.

## Environment

Tested on Google Colab (Python 3.12.12, PyTorch 2.9.0). Create a virtualenv or conda environment before installation.

## File Structure

- `TabDDPM_Aug/` – core implementation (config, data loading, generators, evaluation, experiments)
- `scripts/` – experiment entry points (`run_full_experiment.py`)
- `data/` – input CSV files (`adult.csv`, `pima.csv`, `creditcard.csv`)
- `requirements.txt` – Python dependencies
- `setup.py` – package installation script
- `README.md`, `LICENSE.txt` – documentation and license

Baseline Methods

TabDDPM-Aug is compared against 7 baseline augmentation methods:

**Included via Python dependencies**:
- SMOTE (`imbalanced-learn`)

**Requires separate installation**:
- [CTAB-GAN-Plus](https://github.com/Team-TUD/CTAB-GAN-Plus) - Clone to repo root
- [TabDDPM](https://github.com/yandex-research/tab-ddpm) - Clone to repo root

See [Installation](#installation) for setup instructions.

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

### 3. Install TabDDPM (Required)
```bash
git clone https://github.com/yandex-research/tab-ddpm.git
cd tab-ddpm
pip install -e .
cd ..
```

### 4. Install CTAB-GAN+ (Optional, for full comparisons)
```bash
git clone https://github.com/Team-TUD/CTAB-GAN-Plus.git
```

### 5. Verify Installation
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

**"ModuleNotFoundError: tab_ddpm"**
```bash
git clone https://github.com/yandex-research/tab-ddpm.git
cd tab-ddpm
pip install -e .
```

**"CUDA out of memory"**
- Reduce `batch_size` in `config.py`
- Switch to CPU: `device='cpu'`

**"CTAB-GAN-Plus not found"**
```bash
git clone https://github.com/Team-TUD/CTAB-GAN-Plus.git
```

**Import errors**
```bash
# Ensure proper installation
cd tabddpm-aug
pip install -e .
```

## License

MIT License - see [LICENSE](LICENSE) file.
