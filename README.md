# TabDDPM-Aug: Adaptive Diffusion-Based Hybrid Augmentation for Imbalanced Tabular Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This is the official code for our paper **"TabDDPM-Aug: Adaptive Diffusion-Based Hybrid Augmentation for Imbalanced Tabular Classification"**, currently submitted to *Knowledge-Based Systems*.

## Key Features

- **Adaptive Strategy Selection**: Automatically switches between DBHA (low-density) and FDHA (high-density) regimes.
- **Hybrid Dual-Stream Architecture**: Combines SMOTE-based linear interpolation with an ensemble of K diffusion models for robust minority synthesis.
- **Adaptive Quality Filtering**: Employs IQR-based Distance-to-Closest-Real (DCR) metrics to mitigate distributional collapse and prevent memorization.
- **Privacy-Preserving Synthesis**: Significantly reduces Membership Inference Attack (MIA) risk while maintaining competitive classification utility.
- **Comprehensive Benchmarking**: Validated against 7 baselines across 3 diverse domains using 9 metrics for utility, fidelity, and privacy.

## Environment

Tested on Google Colab (Python 3.12.12, PyTorch 2.9.0). Create a virtualenv or conda environment before installation.

## File Structure

- `TabDDPM_Aug/` – core implementation (config, data loading, generators, evaluation, experiments)
- `scripts/` – experiment entry points (`run_full_experiment.py`, `run_ablation.py`, `run_sensitivity.py`)
- `data/` – input CSV files (`adult.csv`, `pima.csv`, `creditcard.csv`)
- `requirements.txt` – Python dependencies
- `setup.py` – package installation script
- `README.md`, `LICENSE.txt` – documentation and license

Baseline Methods

TabDDPM-Aug is compared against 7 baseline augmentation methods:

**Included via Python dependencies**:
- SMOTE, ADASYN (`imbalanced-learn`)
- CTGAN, TVAE (`ctgan` package)
- GaussianCopula (`sdv` package)

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
# Run the main 9-method comparison
python scripts/run_full_experiment.py --dataset pima

# Ablation study
python scripts/run_ablation.py --dataset adult

# Sensitivity analysis
python scripts/run_sensitivity.py --dataset credit --mode overgen
```

## Datasets

This release includes configurations for three benchmark datasets used in our paper:

- **Adult Income** (`adult.csv`) - UCI Adult dataset
- **Pima Diabetes** (`pima.csv`) - Pima Indians Diabetes
- **Credit Fraud** (`creditcard.csv`) - Credit Card Fraud Detection

**Note**: The framework can be extended to other tabular datasets by:
1. Adding dataset configuration to `config.py`
2. Placing your CSV in `data/` folder with a `target` column
3. Running experiments with `--dataset your_dataset_name`

For dataset details, file locations, and how to add new datasets, see `data/README.md`

## Results

Our TabDDPM-Aug framework improves minority-class classification and offers a better privacy–utility trade-off compared to widely used tabular data augmentation baselines (SMOTE, ADASYN, copula and GAN-based generators, and the original TabDDPM model). All results are averaged over multiple random seeds for stability.

### Adult Income (N=45,222, IR=5.0:1, 5 seeds)

On the Adult Income dataset, TabDDPM-Aug improves F1 while achieving a stronger privacy–utility trade-off than the original TabDDPM and other baselines:

| Method        | F1↑            | AUC↑           | AUPRC↑         | KS↓     | MMD↓    | DCR  | MIA↓    | Time (s) |
|--------------|----------------|----------------|----------------|---------|---------|-------|---------|----------|
| Original     | 0.5036±0.0033  | 0.8579±0.0028  | 0.5054±0.0052  |   –     |   –     |  –    |   –     |   –      |
| SMOTE        | 0.5172±0.0047  | 0.8499±0.0025  | 0.4838±0.0045  |   –     |   –     |  –    |   –     | 0.2      |
| ADASYN       | 0.5110±0.0038  | 0.8482±0.0027  | 0.4807±0.0051  |   –     |   –     |  –    |   –     | 1.0      |
| GaussianCopula | 0.4365±0.0038 | 0.8528±0.0027 | 0.4940±0.0038  | **0.1082** | 0.0149 | 1.30 | 0.9974 | 3.7      |
| CTGAN        | 0.4577±0.0131  | 0.8467±0.0028  | 0.4841±0.0082  | 0.1792 | 0.0257 | 1.81 | 0.9865 | 26.7     |
| TVAE         | 0.4638±0.0101  | 0.8428±0.0077  | 0.4709±0.0156  | 0.2350 | 0.2333 | 0.29 | 0.9965 | 12.8     |
| CTAB-GAN+    | 0.4268±0.0022  | **0.8558±0.0021** | **0.5013±0.0049** | 0.2300 | 0.0327 | 1.95 | 1.0000 | 34.1     |
| TabDDPM      | 0.4196±0.0041  | 0.8535±0.0025  | 0.4972±0.0043  | 0.7586 | 0.8428 | 8.00 | 1.0000 | 19.3     |
| **TabDDPM-Aug** | **0.5208±0.0049** | 0.8504±0.0024 | 0.4861±0.0052 | 0.2247 | **0.0016** | 2.53 | **0.8244** | 46.2 |

For more datasets and a complete discussion of the metrics, please see the paper: **TabDDPM-Aug: Adaptive Diffusion-Based Hybrid Augmentation for Imbalanced Tabular Classification**.


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
