"""
Classification utility metrics for augmented datasets.
"""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats
import os
from .fidelity import compute_ks_statistics, compute_mmd, compute_correlation_distance, calculate_js_divergence, compute_wasserstein_distance, compute_dcr_statistics
from .privacy import membership_inference_attack


def evaluate_comprehensive(X_train, y_train, X_test, y_test, X_synthetic, method_name, seed=42):
    """
    Comprehensive evaluation: utility + fidelity + privacy.

    Args:
        X_train: Training features (real data).
        y_train: Training labels.
        X_test: Test features (real data).
        y_test: Test labels.
        X_synthetic: Synthetic minority samples to evaluate.
        method_name: Name of the generation method (for logging/SDQS).
        seed: Random seed for classifiers.
.
    Returns:
        dict: A collection of metrics including 'f1', 'auc', 'ks_statistic',
              'mia_auc', and 'sdqs'.
    """
    results = {}
    if X_synthetic is None or len(X_synthetic) == 0:
        print("    No synthetic data to evaluate.")
        return { 'f1': 0.0, 'auc': 0.5, 'auprc': 0.0, 'ks_statistic': 1.0, 'mmd': 1.0, 
                 'js_divergence': 1.0, 'wasserstein': 1.0, 'correlation_l2': 10.0, 
                 'mean_dcr': 10.0, 'mia_auc': 0.5, 'sdqs': 0.0 }
                 
    X_minority = X_train[y_train == np.argmin(np.bincount(y_train))]
    if len(X_minority) == 0:
        print("    No minority samples in training data to compare against.")
        X_minority = X_train # Fallback
        
    X_aug = np.vstack([X_train, X_synthetic])
    y_aug = np.hstack([y_train, np.full(len(X_synthetic), np.argmin(np.bincount(y_train)))])
    
    unique, counts = np.unique(y_aug, return_counts=True)
    scale_pos = counts[0] / counts[1] if len(counts) == 2 and counts[1] > 0 else 1.0
    
    catboost_train_dir = '/content/catboost_info/' if os.path.exists('/content/drive') else None
    
    CLASSIFIERS = [
        CatBoostClassifier(iterations=150, learning_rate=0.1, depth=8, scale_pos_weight=min(scale_pos, 40), 
                           random_seed=seed, verbose=False, thread_count=-1, train_dir=catboost_train_dir),
        RandomForestClassifier(n_estimators=100, max_depth=15, random_state=seed, n_jobs=-1),
        XGBClassifier(n_estimators=150, max_depth=8, use_label_encoder=False, eval_metric='logloss', 
                      scale_pos_weight=scale_pos, random_state=seed, n_jobs=-1),
        LogisticRegression(max_iter=500, class_weight='balanced', random_state=seed, n_jobs=-1)
    ]
    
    f1_scores, auc_scores, auprc_scores = [], [], []
    for clf in CLASSIFIERS:
        try:
            clf.fit(X_aug, y_aug)
            y_pred = clf.predict(X_test)
            if hasattr(clf, 'predict_proba'):
                y_proba = clf.predict_proba(X_test)[:, 1]
            else:
                y_proba = clf.decision_function(X_test)
            f1_scores.append(f1_score(y_test, y_pred))
            auc_scores.append(roc_auc_score(y_test, y_proba))
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            auprc_scores.append(auc(recall, precision))
        except Exception as e:
            print(f"    Classifier {type(clf).__name__} failed: {e}")
            f1_scores.append(0.0)
            auc_scores.append(0.5)
            auprc_scores.append(0.0)

    
    results['f1'] = float(np.mean(f1_scores))
    results['auc'] = float(np.mean(auc_scores))
    results['auprc'] = float(np.mean(auprc_scores))
    
    ks_stat, ks_pval = compute_ks_statistics(X_minority, X_synthetic)
    results['ks_statistic'] = float(ks_stat)
    results['ks_pvalue'] = float(ks_pval)
    
    mmd_val = compute_mmd(X_minority[:min(1000, len(X_minority))], X_synthetic[:min(1000, len(X_synthetic))], gamma=0.1)
    results['mmd'] = float(mmd_val)
    
    try: results['js_divergence'] = float(calculate_js_divergence(X_minority, X_synthetic))
    except: results['js_divergence'] = float('nan')
    
    try: results['wasserstein'] = float(compute_wasserstein_distance(X_minority, X_synthetic))
    except: results['wasserstein'] = float('nan')
    
    corr_dist = compute_correlation_distance(X_minority, X_synthetic)
    results['correlation_l2'] = float(corr_dist)
    
    dcr_stats = compute_dcr_statistics(X_minority, X_synthetic)
    results.update(dcr_stats)
    
    mia_auc = membership_inference_attack(X_minority, X_synthetic, X_test)
    results['mia_auc'] = float(mia_auc)
    
    sdqs_results = compute_synthetic_data_quality_score(X_minority, X_synthetic, method_name)
    results.update(sdqs_results)
    
    return results


def evaluate_simple(X_train, y_train, X_test, y_test, method_name, seed=42):
    """
    Simple utility evaluation (F1, AUC, AUPRC only).

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        method_name: Name of the method (unused, kept for symmetry).
        seed: Random seed for classifiers.

    Returns:
        dict: Dictionary with mean F1, AUC, and AUPRC.
    """
    unique, counts = np.unique(y_train, return_counts=True)
    scale_pos = counts[0] / counts[1] if len(counts) == 2 and counts[1] > 0 else 1.0
    
    catboost_train_dir = '/content/catboost_info/' if os.path.exists('/content/drive') else None
    
    CLASSIFIERS = [
        CatBoostClassifier(iterations=150, learning_rate=0.1, depth=8, scale_pos_weight=min(scale_pos, 40), 
                           random_seed=seed, verbose=False, thread_count=-1, train_dir=catboost_train_dir),
        RandomForestClassifier(n_estimators=100, max_depth=15, random_state=seed, n_jobs=-1),
        XGBClassifier(n_estimators=150, max_depth=8, use_label_encoder=False, eval_metric='logloss', 
                      scale_pos_weight=scale_pos, random_state=seed, n_jobs=-1),
        LogisticRegression(max_iter=500, class_weight='balanced', random_state=seed, n_jobs=-1)
    ]
    
    f1_scores, auc_scores, auprc_scores = [], [], []
    for clf in CLASSIFIERS:
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            if hasattr(clf, 'predict_proba'):
                y_proba = clf.predict_proba(X_test)[:, 1]
            else:
                y_proba = clf.decision_function(X_test)
            f1_scores.append(f1_score(y_test, y_pred))
            auc_scores.append(roc_auc_score(y_test, y_proba))
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            auprc_scores.append(auc(recall, precision))
        except Exception as e:
            print(f"    Classifier {type(clf).__name__} failed: {e}")
            f1_scores.append(0.0)
            auc_scores.append(0.5)
            auprc_scores.append(0.0)
    
    return {
        'f1': float(np.mean(f1_scores)),
        'auc': float(np.mean(auc_scores)),
        'auprc': float(np.mean(auprc_scores))
    }


def compute_synthetic_data_quality_score(real_data, synthetic_data, method_name='Unknown'):
    """Composite quality metric combining fidelity, utility, and privacy.

    Args:
        real_data: Real minority samples (reference distribution).
        synthetic_data: Generated synthetic samples.
        method_name: Name of the generation method (for logging).

    Returns:
        dict: SDQS and its components (fidelity_score, utility_score, privacy_score).
    """
    if len(real_data) == 0 or len(synthetic_data) == 0:
        return {'sdqs': 0.0, 'fidelity_score': 0.0, 'utility_score': 0.0, 'privacy_score': 0.0}

    # 1. Distributional Fidelity (inverse KS statistic)
    ks_stats = []
    for i in range(real_data.shape[1]):
        ks_stat, _ = stats.ks_2samp(real_data[:, i], synthetic_data[:, i])
        ks_stats.append(ks_stat)
    fidelity_score = 1.0 - np.nanmean(ks_stats)  # Lower KS = higher fidelity
    
    # 2. Utility Score (TSTR: Train-on-Synthetic-Test-on-Real)
    try:
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        y_synthetic = np.ones(len(synthetic_data))
        y_real = np.ones(len(real_data))
        
        clf.fit(synthetic_data, y_synthetic)
        utility_score = clf.score(real_data, y_real)
    except:
        utility_score = 0.5
    
    # 3. Privacy Score (based on DCR)
    dcr_stats = compute_dcr_statistics(real_data, synthetic_data)
    privacy_score = np.clip(dcr_stats['mean_dcr'] / 5.0, 0, 1) # Normalize DCR
    
    sdqs = 0.3 * fidelity_score + 0.4 * utility_score + 0.3 * privacy_score
    
    print(f"    SDQS for {method_name}: {sdqs:.4f} "
          f"(Fidelity={fidelity_score:.3f}, Utility={utility_score:.3f}, Privacy={privacy_score:.3f})")
    
    return {
        'sdqs': float(sdqs),
        'fidelity_score': float(fidelity_score),
        'utility_score': float(utility_score),
        'privacy_score': float(privacy_score)
    }
