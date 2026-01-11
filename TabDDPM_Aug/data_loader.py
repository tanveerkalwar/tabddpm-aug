"""
Dataset loading and preprocessing utilities.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split

from .config import DATASET_FILES

def load_dataset(dataset_name):
    """Load and preprocess a dataset by name.

    Args:
        dataset_name (str): Key identifying the dataset in DATASET_FILES.

    Returns:
        pandas.DataFrame: Preprocessed dataframe with a binary 'target'
            column and feature columns ready for further processing.
    """
    config = DATASET_FILES[dataset_name]
    for path in [f'data/{config["filename"]}', config["filename"]]:
        if os.path.exists(path):
            df = pd.read_csv(path, skipinitialspace=True)
            break
    else:
        raise FileNotFoundError(f"{config['filename']} not found in {os.getcwd()}")
    
    df.columns = df.columns.str.strip()
    df = df.replace(['?', ' ?', '  ?'], np.nan).dropna()
    
    target_col = config['target_col'] if config['target_col'] in df.columns else df.columns[-1]
    pos_labels = [str(label) for label in config['pos_labels']]
    df['target'] = df[target_col].astype(str).str.strip().apply(lambda x: 1 if x in pos_labels else 0)
    
    if dataset_name == 'credit':
        if 'Time' in df.columns:
            df = df.drop(columns=['Time'])
        if len(df) > 100000:
            print(f"  Subsampling Credit dataset to 50k for efficiency")
            df_maj = df[df['target'] == 0].sample(50000, random_state=42)
            df_min = df[df['target'] == 1]
            df = pd.concat([df_maj, df_min]).sample(frac=1, random_state=42)

    df = df.drop(columns=[target_col] + config['drop_cols'], errors='ignore')
    
    print(f"\n{dataset_name.capitalize()}: {df.shape}, Classes: {df['target'].value_counts().values}")
    return df


def prepare_data(df, seed=42):
    """Split and preprocess dataset for training.

    Args:
        df (pandas.DataFrame): Input dataframe containing features and a binary 'target' column.
        seed (int, optional): Random state for splitting and preprocessing. Defaults to 42.

    Returns:
        dict: Dictionary with normalized train/test arrays, minority
            information, column metadata, and preprocessing objects, including:
            - 'X_train_norm', 'y_train'
            - 'X_test_norm', 'y_test'
            - 'X_minority', 'X_minority_df'
            - 'minority_class', 'n_needed'
            - 'categorical_cols', 'numeric_cols'
            - 'scaler', 'label_encoders', 'X_train_df_full'
    """
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df.drop('target', axis=1), df['target'], 
        test_size=0.2, random_state=seed, stratify=df['target']
    )
    
    unique, counts = np.unique(y_train, return_counts=True)
    if len(counts) < 2:
        print("Warning: Only one class found in training data.")
        minority_class = 0 # Default
    else:
        minority_class = unique[np.argmin(counts)]
    
    numeric_cols = list(X_train_df.select_dtypes(include=np.number).columns)
    categorical_cols = list(X_train_df.select_dtypes(include='object').columns)
    
    # Use QuantileTransformer for robustness to outliers (like in 'Amount')
    scaler = QuantileTransformer(output_distribution='uniform', random_state=seed)
    
    # Handle empty numeric cols
    if not numeric_cols:
        X_train_norm_num = np.empty((len(X_train_df), 0))
        X_test_norm_num = np.empty((len(X_test_df), 0))
    else:
        X_train_norm_num = scaler.fit_transform(X_train_df[numeric_cols])
        X_test_norm_num = scaler.transform(X_test_df[numeric_cols])
    
    X_train_processed, X_test_processed, label_encoders = [], [], {}
    for col in categorical_cols:
        le = LabelEncoder()
        all_values = pd.concat([X_train_df[col], X_test_df[col]]).astype(str).unique()
        le.fit(all_values)
        X_train_processed.append(le.transform(X_train_df[col].astype(str)).reshape(-1, 1))
        X_test_processed.append(le.transform(X_test_df[col].astype(str)).reshape(-1, 1))
        label_encoders[col] = le
    
    if X_train_processed:
        X_train_norm = np.hstack([X_train_norm_num, np.hstack(X_train_processed)])
        X_test_norm = np.hstack([X_test_norm_num, np.hstack(X_test_processed)])
    else:
        X_train_norm, X_test_norm = X_train_norm_num, X_test_norm_num
    
    X_minority = X_train_norm[y_train.values == minority_class]
    X_minority_df = X_train_df[y_train.values == minority_class].reset_index(drop=True)
    
    n_needed = 0
    if len(counts) == 2:
        n_needed = counts.max() - counts.min()
    
    return {
        'X_train_norm': X_train_norm, 'y_train': y_train.values,
        'X_test_norm': X_test_norm, 'y_test': y_test.values,
        'X_minority': X_minority, 'X_minority_df': X_minority_df,
        'minority_class': minority_class, 'n_needed': n_needed,
        'categorical_cols': categorical_cols, 'numeric_cols': numeric_cols,
        'scaler': scaler, 'label_encoders': label_encoders, 'X_train_df_full': X_train_df
    }
