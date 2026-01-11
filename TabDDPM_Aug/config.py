"""
Configuration constants for TabDDPM-Aug experiments.
"""

DATASET_CONFIGS = {
    'adult': {
        'tabddpm_epochs': 300,
        'batch_size': 512,
        'lr': 1e-4,
        'n_seeds': 5,
    },
    'pima': {
        'tabddpm_epochs': 600,
        'batch_size': 128,
        'lr': 5e-4,
        'n_seeds': 5,
    },
    'credit': {
        'tabddpm_epochs': 400,
        'batch_size': 128,
        'lr': 1e-4,
        'n_seeds': 5,
    }
}


def get_config(dataset_name):
    """Retrieve configuration for a dataset.

    Args:
        dataset_name (str): Dataset key, e.g. 'adult', 'pima', or 'credit'.

    Returns:
        dict: Hyperparameter configuration for the given dataset.

    Raises:
        KeyError: If the dataset name is not in DATASET_CONFIGS.
    """
    return DATASET_CONFIGS[dataset_name]

# Dataset file names and label mappings used by load_dataset
DATASET_FILES = {
    'adult': {
        'filename': 'adult.csv',
        'target_col': 'income',
        'pos_labels': ['>50K', '1'],
        'drop_cols': []
    },
    'pima': {
        'filename': 'pima.csv',
        'target_col': 'Outcome',
        'pos_labels': [1],
        'drop_cols': []
    },
    'credit': {
        'filename': 'creditcard.csv',
        'target_col': 'default',
        'pos_labels': [1],
        'drop_cols': []
    }
}
