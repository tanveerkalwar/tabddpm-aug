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
        'tabddpm_epochs': 800,
        'batch_size': 128,
        'lr': 5e-4,
        'n_seeds': 5,
    },
    'credit': {
        'tabddpm_epochs': 400,
        'batch_size': 128,
        'lr': 1e-4,
        'n_seeds': 5,
    },
    'letter_recognition': {
        'tabddpm_epochs': 1000,
        'batch_size': 128,
        'lr': 5e-5,
        'n_seeds': 5,
    },
    'kc1': {
        'tabddpm_epochs': 600,
        'batch_size': 128,
        'lr': 5e-4,
        'n_seeds': 5,
    },
    'pc4': {
        'tabddpm_epochs': 800,
        'batch_size': 128,
        'lr': 1e-4,
        'n_seeds': 5,
    },
    'ecoli': {
        'n_seeds': 5,
        'tabddpm_epochs': 600,
        'batch_size': 128,
        'lr': 5e-4,
    },
    'magic': {
        'n_seeds': 5, 
        'tabddpm_epochs': 300, 
        'batch_size': 256, 
        'lr': 1e-4
    },
    'covertype': {
        'n_seeds': 5, 
        'tabddpm_epochs': 200, 
        'batch_size': 1024, 
        'lr': 1e-4},
    'jm1': {
        'n_seeds': 5,
        'tabddpm_epochs': 800,
        'batch_size': 64,
        'lr': 2e-4,
    },
    'pc3': {
        'n_seeds': 5,
        'tabddpm_epochs': 800,
        'batch_size': 64,
        'lr': 2e-4,
    },
    'kc2': {
        'n_seeds': 5,
        'tabddpm_epochs': 800,
        'batch_size': 64,
        'lr': 2e-4,
    },
    'taiwanese': {
        'n_seeds': 5,
        'tabddpm_epochs': 800,
        'batch_size': 128,
        'lr': 5e-5,
    },
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
    },
    'letter_recognition': {
        'filename': 'letter.csv',
        'target_col': 'letter',
        'pos_labels': ['Z'],
        'drop_cols': []
    },
    'kc1': {
        'filename': 'kc1.csv',
        'target_col': 'target',
        'pos_labels': [1, '1'],
        'drop_cols': []
    },
    'pc4': {
        'filename': 'pc4.csv',
        'target_col': 'target',
        'pos_labels': [1, '1'],
        'drop_cols': []
    },
    'ecoli': {
        'filename': 'ecoli.csv',
        'target_col': 'class',
        'pos_labels': ['im'],
        'drop_cols': []
    },
   'magic': {
       'filename': 'magic.csv', 
       'target_col': 'class', 
       'pos_labels': ['g'], 
       'drop_cols': []
   },
    'covertype': {
        'filename': 'covertype.csv', 
        'target_col': 'Cover_Type', 
        'pos_labels': [4], 
        'drop_cols': []
    },
    'jm1': {
        'filename': 'jm1.csv', 
        'target_col': 'defects', 
        'pos_labels': ['true'], 
        'drop_cols': []
    },
    'pc3': {
        'filename': 'pc3.csv',
        'target_col': 'c',
        'pos_labels': [1, '1'],
        'drop_cols': []
    },
    'kc2': {
        'filename': 'kc2.csv',
        'target_col': 'problems',
        'pos_labels': [1, '1'],
        'drop_cols': []
    },
    'taiwanese': {
        'filename': 'taiwanese.csv',
        'target_col': 'default.payment.next.month',
        'pos_labels': [1],
        'drop_cols': [],
    },
}
