# Datasets

In our experiments, we used preprocessed **CSV files** (`adult.csv`, `pima.csv`, `creditcard.csv`). You can adapt the code to whatever tabular format is convenient for you (CSV, TSV, parquet, etc.) by modifying the dataset loading logic.

By default, the script expects:

- Files listed in `DATASET_FILES` (e.g., `adult.csv`)
- A binary target column specified by `target_col` and `pos_labels`

Place your CSV files here, each with a binary label column specified by target_col in DATASET_FILES.

## Benchmark Datasets (from paper)

### 1. Adult Income
- **Source**: https://archive.ics.uci.edu/dataset/2/adult
- **Target**: `income` (>50K=1, ≤50K=0)
- **Description**: Census data with 48,842 samples, 14 features

### 2. Pima Diabetes  
- **Source**: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **Target**: `Outcome` (diabetic=1, healthy=0)
- **Description**: Medical data with 768 samples, 8 features

### 3. Credit Card Fraud
- **Source**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Due to size constraints, `creditcard.csv` is not included in this repository.
- Please download the original dataset from Kaggle, save it as `data/creditcard.csv`, and ensure the filename matches `DATASET_FILES['credit']['filename']` in `config.py`.
- **Target**: `default` (fraud=1, legitimate=0)
- **Description**: Transaction data with 284,807 samples, 29 features

## Using Custom Datasets

1. Prepare a CSV with a binary target column and specify its name via `target_col` in `config.py`
2. Add configuration to `TabDDPM_Aug/config.py`:
```python
DATASET_FILES['your_dataset'] = {
    'filename': 'your_data.csv',
    'target_col': 'label',  # Your target column name
    'pos_labels': [1],      # Positive class values
    'drop_cols': []         # Columns to ignore
}

DATASET_CONFIGS['your_dataset'] = {
    'tabddpm_epochs': 500, # more training if needed
    'batch_size': 256, # adjust to your GPU memory
    'lr': 2e-4, # tune learning rate
    'n_seeds': 5 # fewer seeds to run faster (runs)
}
```

3. Run experiments:
```bash
python scripts/run_full_experiment.py --dataset your_dataset
```