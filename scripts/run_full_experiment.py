#!/usr/bin/env python3
"""
Usage:
    python scripts/run_full_experiment.py --dataset pima
"""
import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import time
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from TabDDPM_Aug.data_loader import load_dataset, prepare_data
from TabDDPM_Aug.config import get_config
from TabDDPM_Aug.generators.tabddpm_aug import tabddpm_aug_final
from TabDDPM_Aug.evaluation.utility import evaluate_simple, evaluate_comprehensive


def run_full_experiment(df, device, main_config, dataset_name):
    """Execute TabDDPM_Aug.

    Args:
        df (pandas.DataFrame): Preprocessed dataset containing features and a binary 'target' column.
        device (str or torch.device): Computation device, e.g. "cpu" or "cuda".
        main_config (dict): Dataset-specific hyperparameters, including 'n_seeds' and TabDDPM settings.
        dataset_name (str): Dataset identifier used for logging and summaries.

    Returns:
        None: aggregated table.
    """

    n_seeds = main_config['n_seeds']
    print(f"Running {n_seeds} seeds for statistical significance\n")
    
    results = {
        'tabddpm_aug': []
    }
    timing_info = defaultdict(list)
    
    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx * 100
        print(f"Seed {seed_idx+1}/{n_seeds} (seed={seed})")
        
        data = prepare_data(df, seed)
        
        if data['n_needed'] <= 0:
            print("Dataset is balanced, skipping seed.")
            continue
        
        #TabDDPM-Aug        
        try:
            start_t = time.time()
            X_tabddpm_aug = tabddpm_aug_final(
                data['X_train_norm'], data['y_train'], main_config, seed, device
            )
            elapsed = time.time() - start_t
            timing_info['tabddpm_aug'].append(elapsed)
            
            if X_tabddpm_aug is not None and len(X_tabddpm_aug) > 0:
                r = evaluate_comprehensive(data['X_train_norm'], data['y_train'], 
                                           data['X_test_norm'], data['y_test'], 
                                           X_tabddpm_aug, 'TabDDPM-Aug', seed)
                results['tabddpm_aug'].append(r)
            else:
                print("  TabDDPM-Aug failed to generate samples")
        except Exception as e:
            print(f"  FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print Final Full Experiment Summary
    print("FINAL RESULTS - Classification Performance")
    print(f"\n{'Method':<20} {'F1':<18}{'Time (s)':<12}")
    print("-"*100)
    
    for method in ['tabddpm_aug']:
        f1_vals = [r.get('f1', np.nan) for r in results[method]]
        if not f1_vals:
            continue
        time_str = f"{np.mean(timing_info[method]):.1f}" if method in timing_info else "N/A"
        print(f"{method:<20} {np.nanmean(f1_vals):.4f}±{np.nanstd(f1_vals):.4f}  "
              f"{time_str:<12}")
    
    print("FINAL RESULTS - Statistical Fidelity & Privacy")
    print(f"\n{'Method':<20} {'KS ↓':<10} {'MMD ↓':<10} {'DCR':<10} {'MIA ↓':<10}")
    print("-"*120)
    
    for method in ['tabddpm_aug']:
        if method not in results or not results[method] or 'ks_statistic' not in results[method][0]:
            continue
        ks_vals = [r.get('ks_statistic', np.nan) for r in results[method]]
        mmd_vals = [r.get('mmd', np.nan) for r in results[method]]
        dcr_vals = [r.get('mean_dcr', np.nan) for r in results[method]]
        mia_vals = [r.get('mia_auc', np.nan) for r in results[method]]
        
        print(f"{method:<20} {np.nanmean(ks_vals):.4f}   {np.nanmean(mmd_vals):.4f}   {np.nanmean(dcr_vals):.2f}   {np.nanmean(mia_vals):.4f}")
            
def main():
    parser = argparse.ArgumentParser(description='Run full 9-method comparison')
    # Add the new dataset identifiers to the choices list
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['adult', 'pima', 'credit', 'letter_recognition','kc1', 'pc4', 'ecoli', 'magic', 'covertype','jm1', 'pc3', 'kc2','taiwanese'],
                        help='Dataset to use for experiments')
    args = parser.parse_args()

    print(f"Dataset: {args.dataset}")
    
    df = load_dataset(args.dataset)
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        device = 'cpu'
    print(f"Device: {device}\n")
    
    main_config = get_config(args.dataset)
    
    run_full_experiment(df, device, main_config, args.dataset)
    
    print("EXPERIMENT COMPLETE")


if __name__ == '__main__':
    main()
