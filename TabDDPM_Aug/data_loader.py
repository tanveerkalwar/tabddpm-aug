"""
Dataset loading and preprocessing utilities.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.datasets import fetch_openml
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
    df = None

    if dataset_name == 'letter_recognition':
        data = fetch_openml(name='letter', version=1, as_frame=True, parser='liac-arff')
        df = data.frame
        target_col = data.target.name
        df['target'] = (df[target_col] == 'Z').astype(int)
        df = df.drop(columns=[target_col])

    elif dataset_name == 'kc1':
        raw = fetch_openml(name='kc1', version=1, as_frame=True, parser='liac-arff')
        df = raw.frame.copy()
        df = df.replace('?', np.nan).dropna()
        target_col_raw = [c for c in df.columns if 'defect' in c.lower() or c == raw.target.name]
        tname = target_col_raw[0] if target_col_raw else df.columns[-1]
        df['target'] = df[tname].astype(str).str.lower().apply(
            lambda x: 1 if x in ['true', '1', 'yes', 'y'] else 0
        )
        df = df.drop(columns=[tname])

    elif dataset_name == 'pc4':
        raw = fetch_openml(name='PC4', version=1, as_frame=True, parser='liac-arff')
        df = raw.frame.copy()
        df = df.replace('?', np.nan).dropna()
        target_col_raw = [c for c in df.columns if 'defect' in c.lower() or c == raw.target.name]
        tname = target_col_raw[0] if target_col_raw else df.columns[-1]
        df['target'] = df[tname].astype(str).str.lower().apply(
            lambda x: 1 if x in ['true', '1', 'yes', 'y'] else 0
        )
        df = df.drop(columns=[tname])
        target_col = 'target'

    elif dataset_name == 'ecoli':
        data = fetch_openml(name='ecoli', version=1, as_frame=True, parser='liac-arff')
        df = data.frame
        target_col = data.target.name
        df['target'] = (df[target_col] == 'im').astype(int)
        df = df.drop(columns=[target_col])

    elif dataset_name == 'magic':
        data = fetch_openml(name='magic', version=1, as_frame=True, parser='liac-arff')
        df = data.frame
        target_col = data.target.name
        df['target'] = (df[target_col].astype(str) == '1').astype(int)
        df = df.drop(columns=[target_col])

    elif dataset_name == 'covertype':
        data = fetch_openml(name='covertype', version=1, as_frame=True, parser='liac-arff')
        df = data.frame
        target_col = data.target.name
        df['target'] = (df[target_col].astype(str) == 'Cottonwood_Willow').astype(int)
        df = df.drop(columns=[target_col])

    elif dataset_name == 'jm1':
        data = fetch_openml(data_id=1053, as_frame=True, parser='liac-arff')
        df = data.frame
        target_col = data.target.name
        # Binary defect dataset: values 'true'/'false' or 1/0. Convert to int.
        df['target'] = (df[target_col].astype(str) == 'true').astype(int)
        df = df.drop(columns=[target_col])

    elif dataset_name == 'pc3':
        raw = fetch_openml(data_id=1050, as_frame=True, parser='liac-arff')
        df = raw.frame
        target_col = raw.target.name   # 'c'
        # Values: 'FALSE' (0) and 'TRUE' (1) – minority is 'TRUE'
        df['target'] = (df[target_col].astype(str) == 'TRUE').astype(int)
        df = df.drop(columns=[target_col])
        target_col = 'target'

    elif dataset_name == 'kc2':
        raw = fetch_openml(data_id=1063, as_frame=True, parser='liac-arff')
        df = raw.frame
        target_col = raw.target.name   # 'problems'
        # Values: 'no' (0) and 'yes' (1) – minority is 'yes'
        df['target'] = (df[target_col].astype(str) == 'yes').astype(int)
        df = df.drop(columns=[target_col])
        target_col = 'target'

    elif dataset_name == 'taiwanese':
        data = fetch_openml(data_id=42477, as_frame=True, parser='liac-arff')
        df = data.frame
        target_col = data.target.name  # 'default.payment.next.month'
        # default = 1 (minority)
        df['target'] = df[target_col].astype(int)
        df = df.drop(columns=[target_col])
        target_col = 'target'

    else:
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

    cols_to_drop = [c for c in [target_col] + config.get('drop_cols', []) if c in df.columns and c != 'target']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Ensure target column is named 'target' and is integer
    if 'target' not in df.columns:
        raise ValueError("No 'target' column found after preprocessing")
    df['target'] = df['target'].astype(int)

    # Convert any non‑numeric (object/category) columns to numeric codes
    for col in df.columns:
        if col == 'target':
            continue
        if df[col].dtype.name == 'category' or df[col].dtype == 'object':
            # Convert to category first, then codes
            df[col] = df[col].astype('category').cat.codes
        # Ensure numeric (might be float/int already)
        if not np.issubdtype(df[col].dtype, np.number):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass

    # Drop columns that are constant (zero variance) – but keep at least one feature
    constant_cols = []
    for col in df.columns:
        if col != 'target' and df[col].nunique() <= 1:
            constant_cols.append(col)
    if constant_cols:
        print(f"  Dropping constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)

    # After dropping, ensure at least one feature remains
    if len(df.columns) == 1:   # only 'target' left
        raise ValueError(f"Dataset {dataset_name} has no features after dropping constant columns. Skipping.")
        
    if df.isnull().any().any():
        print(f"  Warning: Dropping {df.isnull().sum().sum()} missing values")
        df = df.dropna()

    if len(df) == 0:
        raise ValueError(f"Dataset {dataset_name} has zero samples after cleaning. Check data source or imputation.")

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
