"""
TabDDPM-Aug: Adaptive Hybrid Augmentation Framework

Implements:
- DCR (Distance-to-Closest-Real) filtering with IQR-based thresholds
- DBHA (Difficulty-Based Hybrid Augmentation) for low-density regimes
- FDHA (Full-Data Hybrid Augmentation) for high-density regimes
- Ensemble generation for robustness
"""
import numpy as np
import torch
import gc
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

try:
    from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
    from tab_ddpm.modules import MLPDiffusion
    TABDDPM_AVAILABLE = True
except ImportError:
    TABDDPM_AVAILABLE = False



def dcr_filtering(X_synthetic, X_real, n_needed, seed=42):
    """
    Filter synthetic candidates using Distance-to-Closest-Real with IQR thresholds.
    Applies IQR-based outlier detection (Tukey fences) to balance fidelity 
    and novelty. Rejects memorized samples (DCR < Q1 - 0.5×IQR) and extreme 
    outliers (DCR > Q3 + 1.5×IQR).
    
    Args:
        X_synthetic: Candidate pool, shape (n_candidates, n_features)
        X_real: Training samples for distance reference
        n_needed: Target count after filtering
        seed: Random state for reproducible selection
    
    Returns:
        ndarray: Filtered samples, shape (n_needed, n_features)
    """
    if len(X_synthetic) == 0:
        print(f"    Empty synthetic pool received!")
        return X_synthetic
    
    print(f"    DCR Filtering {len(X_synthetic)} candidates → target {n_needed}...")
    
    # Compute distances to nearest real sample
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    nn.fit(X_real)
    distances, _ = nn.kneighbors(X_synthetic)
    dcr_values = distances.flatten()
    
    # Statistics
    median_dcr = np.median(dcr_values)
    q25_dcr = np.percentile(dcr_values, 25)
    q75_dcr = np.percentile(dcr_values, 75)
    
    print(f"    DCR Stats: Q1={q25_dcr:.3f}, Median={median_dcr:.3f}, Q3={q75_dcr:.3f}")
    
    # ========================================================================
    # IQR-based thresholds (Tukey-style fences)
    # ========================================================================
    iqr = q75_dcr - q25_dcr # IQR (interquartile range)
    lower_bound = q25_dcr - 0.5 * iqr # Reject memorized samples
    upper_bound = q75_dcr + 1.5 * iqr # Reject extreme outliers
    
    print(f"    IQR-based bounds: [{lower_bound:.3f}, {upper_bound:.3f}] (IQR={iqr:.3f})")
    
    # Filter: keep samples within bounds
    valid_mask = (dcr_values >= lower_bound) & (dcr_values <= upper_bound)
    X_filtered = X_synthetic[valid_mask]
    
    n_filtered = len(X_filtered)
    n_removed = len(X_synthetic) - n_filtered
    removal_pct = 100 * n_removed / len(X_synthetic)
    
    print(f"    Filtered: {len(X_synthetic)} → {n_filtered} samples (removed {removal_pct:.1f}%)")
    
    np.random.seed(seed)
    
    # ========================================================================
    # CASE 1: Normal - plenty of valid candidates after filtering
    # ========================================================================
    if n_filtered >= n_needed:
        print(f"    CASE 1: Normal selection from {n_filtered} candidates")
        indices = np.random.choice(n_filtered, size=n_needed, replace=False)
        X_final = X_filtered[indices]
        return X_final
    
    # ========================================================================
    # CASE 2: All filtered out - use closest-to-median as best fallback
    # ========================================================================
    elif n_filtered == 0:
        print(f"    CASE 2: All samples filtered! Using closest-to-median fallback...")
        dcr_distance_from_median = np.abs(dcr_values - median_dcr)
        best_indices = np.argsort(dcr_distance_from_median)[:n_needed]
        X_final = X_synthetic[best_indices]
        print(f"    Selected {len(X_final)} samples closest to median DCR={median_dcr:.3f}")
        return X_final
    
    # ========================================================================
    # CASE 3: Pool too small but non-empty - use all valid + augment
    # ========================================================================
    else:
        shortage = n_needed - n_filtered
        shortage_pct = 100 * shortage / n_needed
        
        print(f"    CASE 3: Only {n_filtered} valid samples, need {n_needed}. "
              f"Short by {shortage} ({shortage_pct:.1f}%)")
        
        if shortage > n_filtered:
            print(f"    WARNING: Severe shortage! Will have {100*shortage/n_needed:.0f}% duplicates")
        
        # Use all valid samples + resample with replacement for shortage
        additional_indices = np.random.choice(
            n_filtered, 
            size=shortage, 
            replace=True
        )
        X_augmented = X_filtered[additional_indices]
        X_final = np.vstack([X_filtered, X_augmented])
        
        print(f"    Combined: {n_filtered} unique + {shortage} resampled = {len(X_final)} total")
        return X_final

def tabddpm_aug_ensemble_generator(X_minority_norm, n_needed, config, seed=42, device='cpu', n_models=3, overgen_factor=1.8):
    """
    NON-conditional ensemble TabDDPM generator with flexible overgeneration.

    Args:
        X_minority_norm (numpy.ndarray): Normalized minority-class samples, shape (n_minority, n_features).
        n_needed (int): Target number of samples to select after filtering.
        config (dict): Training hyperparameters (e.g., 'lr', 'tabddpm_epochs', 'batch_size').
        seed (int, optional): Base random seed for NumPy and PyTorch.
        device (str or torch.device, optional): Computation device, e.g. 'cpu' or 'cuda'.
        n_models (int, optional): Number of ensemble diffusion models.
        overgen_factor (float, optional): Multiplier for overgeneration before DCR filtering.

    Returns:
        numpy.ndarray: Overgenerated candidate pool of synthetic samples,
            shape (n_candidates, n_features). Returns an empty array with
            shape (0, n_features) if no models could be trained.
    """
    if not TABDDPM_AVAILABLE:
        raise RuntimeError("TabDDPM not available")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"    Ensemble TabDDPM: Training {n_models} models (overgen={overgen_factor}×)...")
    
    # Calculate per-model generation count
    # Total = n_needed * overgen_factor
    # Per model = Total / n_models
    n_per_model = max(1, int(n_needed * overgen_factor / n_models))
    total_candidates = n_per_model * n_models
    
    print(f"      Target: {n_needed} → Pool: {total_candidates} ({overgen_factor}×)")
    
    all_samples = []
    
    for model_idx in range(n_models):
        model_seed = seed + model_idx * 1000
        
        print(f"        Model {model_idx+1}/{n_models} (seed={model_seed}, n_gen={n_per_model})...")
        
        np.random.seed(model_seed)
        torch.manual_seed(model_seed)
        
        n_features = X_minority_norm.shape[1]
        
        model = MLPDiffusion(
            d_in=n_features, 
            num_classes=0, 
            is_y_cond=False,
            rtdl_params={
                'd_in': n_features, 
                'd_layers': [256, 256],
                'd_out': n_features, 
                'dropout': 0.0
            },
            dim_t=128
        ).to(device)
        
        diffusion = GaussianMultinomialDiffusion(
            num_classes=np.array([0]), 
            num_numerical_features=n_features,
            denoise_fn=model, 
            num_timesteps=1000, 
            gaussian_loss_type='mse',
            scheduler='cosine', 
            device=device
        ).to(device)
        
        # Adaptive learning rate: ±10% variation across models for diversity
        lr_multiplier = 1.0 + (model_idx - n_models//2) * 0.1
        lr = config.get('lr', 2e-4) * 0.8 * lr_multiplier
        
        optimizer = torch.optim.AdamW(
            diffusion.parameters(), 
            lr=lr, 
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.get('tabddpm_epochs', 600),
            eta_min=lr * 0.05
        )
        
        diffusion.train()
        X_tensor = torch.FloatTensor(X_minority_norm).to(device)
        batch_size = min(config.get('batch_size', 256), len(X_tensor))
        
        if batch_size == 0:
            print("        Skipping model, not enough data.")
            continue
            
        final_loss = 0.0
        
        for epoch in range(config.get('tabddpm_epochs', 600)):
            perm = torch.randperm(len(X_tensor))
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(X_tensor), batch_size):
                indices = perm[i:i+batch_size]
                x_batch = X_tensor[indices]
                
                optimizer.zero_grad()
                out_dict = {}
                loss_multi, loss_gauss = diffusion.mixed_loss(x_batch, out_dict)
                loss = loss_multi.mean() + loss_gauss.mean()
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            scheduler.step()
            final_loss = epoch_loss / (n_batches if n_batches > 0 else 1)
        
        # Generate samples
        diffusion.eval()
        with torch.no_grad():
            y_dist = torch.ones(n_per_model, 1, dtype=torch.float32, device=device)
            sample_output = diffusion.sample(n_per_model, y_dist=y_dist)
            
            if isinstance(sample_output, tuple):
                for candidate in sample_output:
                    if isinstance(candidate, torch.Tensor) and candidate.shape == (n_per_model, n_features):
                        batch_samples = candidate
                        break
            else:
                batch_samples = sample_output
            
            samples_np = batch_samples.cpu().numpy()
            samples_np = np.clip(samples_np, 0, 1)
            all_samples.append(samples_np)
        
        print(f"        Model {model_idx+1} complete. Clearing cache...")
        del model
        del diffusion
        torch.cuda.empty_cache()
        gc.collect()  
    
    if not all_samples:
        print("        Ensemble generation failed, returning no samples.")
        return np.array([]).reshape(0, X_minority_norm.shape[1])

    X_pool = np.vstack(all_samples)
    print(f"      Generated diverse pool: {len(X_pool)} samples from {n_models} models")
    
    return X_pool


def find_hard_samples(X_train, y_train, X_minority, minority_class, seed=42):
    """
    Split minority samples into hard and easy subsets based on classification confidence.
    
    Uses logistic regression to identify samples with low predicted probability
    (< 0.7) as "hard" examples that benefit from diffusion-based synthesis.
    
    Args:
        X_train: Full training set for classifier fitting
        y_train: Training labels
        X_minority: Minority class samples to partition
        minority_class: Label of minority class
        seed: Random state
    
    Returns:
        tuple: (X_minority_hard, X_minority_easy)
    """
    print("    Finding 'hard' vs 'easy' minority samples...")
    clf = LogisticRegression(class_weight='balanced', random_state=seed, max_iter=300, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred_proba_minority = clf.predict_proba(X_minority)[:, minority_class]
    
    hardness_threshold = 0.7 
    hard_mask = (y_pred_proba_minority < hardness_threshold)
    
    X_minority_hard = X_minority[hard_mask]
    X_minority_easy = X_minority[~hard_mask]
    
    if len(X_minority_hard) < 2: # Need at least 2 samples for ensemble
        print("    WARNING: No hard samples found, using 20% random split.")
        X_minority_hard = X_minority[:max(2, int(len(X_minority)*0.2))]
        X_minority_easy = X_minority[max(2, int(len(X_minority)*0.2)):]
    if len(X_minority_easy) < 2: # Need at least 2 samples for SMOTE
        X_minority_easy = X_minority
        
    print(f"    Split: {len(X_minority_hard)} hard samples, {len(X_minority_easy)} easy samples.")
    
    return X_minority_hard, X_minority_easy


def tabddpm_aug_final(X_train, y_train, config, seed=42, device="cpu"):
    """
    Main adaptive augmentation strategy (ADAPTIVE-DBHA-DCR).

    Automatically selects:
    - DBHA for low-density regimes (ρ < 30)
    - FDHA for high-density regimes (ρ ≥ 30)

    Args:
        X_train (numpy.ndarray): Full training feature matrix, shape (n_samples, n_features).
        y_train (numpy.ndarray): Training labels, shape (n_samples,).
        config (dict): Hyperparameters for TabDDPM and SMOTE (e.g., 'lr', 'tabddpm_epochs', 'batch_size').
        seed (int, optional): Random seed for NumPy and PyTorch. Defaults to 42.
        device (str or torch.device, optional): Device for diffusion training and sampling, e.g. "cpu" or "cuda".

    Returns:
        numpy.ndarray or None: Final augmented minority samples of shape
            (n_needed, n_features), or None if augmentation is skipped
            or fails (e.g., single-class dataset, no minority, etc.).
    """
    print(f"\n[TabDDPM-Aug: Adaptive-DBHA-DCR (Ensemble + DCR Filter)]")
    
    # Get dataset characteristics
    unique, counts = np.unique(y_train, return_counts=True)
    
    if len(counts) < 2:
        print("  Dataset is single-class. Skipping augmentation.")
        return None
        
    minority_class = unique[np.argmin(counts)]
    majority_class = unique[np.argmax(counts)]
    
    n_needed = 0
    if len(counts) == 2:
        n_needed = counts.max() - counts.min()
    if n_needed <= 0:
        print("  Dataset is already balanced or has no minority. Skipping.")
        return None
        
    X_minority = X_train[y_train == minority_class]
    X_majority = X_train[y_train == majority_class]
    
    n_minority = len(X_minority)
    if n_minority == 0:
        print("  No minority samples found. Skipping.")
        return None
        
    imbalance_ratio = counts.max() / (counts.min() + 1e-6)
    
    print(f"  Minority: {n_minority} samples")
    print(f"  Target: {n_needed} synthetic samples")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    # ========================================================================
    # STEP 1: ADAPTIVE SPLIT STRATEGY
    # ========================================================================

    # Dataset characteristics: sample-to-feature ratio ρ = N_min / d
    d = X_train.shape[1]
    rho = n_minority / max(d, 1)
    print(f"Features d = {d}, density ratio ρ = {rho:.2f}")

    tau_size = 30.0  # density threshold used to separate low vs high-density regimes.

    if rho < tau_size:  # Low-density → DBHA
        print("Strategy: Low-density regime (DBHA). Using Easy–Hard split.")
        X_minority_hard, X_minority_easy = find_hard_samples(
            X_train, y_train, X_minority, minority_class, seed
        )
    else:  # High-density → FDHA
        print("Strategy: High-density regime (FDHA). Using full minority data.")
        X_minority_hard = X_minority
        X_minority_easy = X_minority


    # ========================================================================
    # ADAPTIVE RATIO SELECTION
    # ========================================================================
    if imbalance_ratio > 100: # Extreme (like Credit)
        print(f"  Ratio: EXTREME imbalance (90% SMOTE + 10% TabDDPM)")
        smote_ratio = 0.90
    elif n_minority < 500: # Small (like Pima)
        print(f"  Ratio: SMALL dataset (50% SMOTE + 50% TabDDPM)")
        smote_ratio = 0.50
    else: # Medium/Large (like Adult)
        print(f"  Ratio: MEDIUM dataset (70% SMOTE + 30% TabDDPM)")
        smote_ratio = 0.70
    
    n_smote = int(n_needed * smote_ratio)
    n_tabddpm = n_needed - n_smote
    
    # ========================================================================
    # STEP 2: SMOTE Generation (on "Easy" or "All" samples)
    # ========================================================================
    print(f"\n  Step 2: Generating {n_smote} SMOTE samples from {len(X_minority_easy)} samples...")
    X_smote = np.array([]).reshape(0, X_minority.shape[1])
    if SMOTE_AVAILABLE and n_smote > 0 and len(X_minority_easy) > 1:
        try:
            X_temp = np.vstack([X_minority_easy, X_majority[:min(len(X_majority), len(X_minority_easy)*2)]])
            y_temp = np.hstack([
                np.full(len(X_minority_easy), minority_class),
                np.full(min(len(X_majority), len(X_minority_easy)*2), majority_class)
            ])
            
            smote = SMOTE(
                sampling_strategy={minority_class: len(X_minority_easy) + n_smote},
                k_neighbors=min(5, len(X_minority_easy) - 1),
                random_state=seed
            )
            X_resampled, y_resampled = smote.fit_resample(X_temp, y_temp)
            X_smote = X_resampled[len(X_temp):len(X_temp) + n_smote]
            print(f"    Generated {len(X_smote)} SMOTE samples")
            
        except Exception as e:
            print(f"    SMOTE failed: {e}. Defaulting to random samples.")
            indices = np.random.choice(len(X_minority_easy), n_smote, replace=True)
            X_smote = X_minority_easy[indices]
    elif n_smote > 0:
        print("    Skipped SMOTE (not enough samples)")

    # ========================================================================
    # STEP 3: Ensemble + DCR Generation (on "Hard" or "All" samples)
    # ========================================================================
    X_tabddpm_final = np.array([]).reshape(0, X_minority.shape[1])
    if n_tabddpm > 0 and len(X_minority_hard) > 1:
        print(f"\n  Step 3: Generating diverse pool from {len(X_minority_hard)} samples...")
        
        try:
            # 1. Generate DIVERSE pool using Ensemble
            X_tabddpm_pool = tabddpm_aug_ensemble_generator(
                X_minority_hard,
                n_tabddpm, # Pass n_needed, ensemble will over-generate
                config, seed, device, n_models=3
            )
            
            if X_tabddpm_pool is not None and len(X_tabddpm_pool) > 0:
                print(f"  Step 4: DCR Filtering (Fidelity/Privacy Filter)...")
                # 2. Filter for QUALITY/NOVELTY using DCR
                X_tabddpm_final = dcr_filtering(
                    X_tabddpm_pool,
                    X_minority, # Filter against all real minority samples
                    n_tabddpm,
                    seed=seed
                )
                print(f"    Selected {len(X_tabddpm_final)} high-quality samples")
                
        except Exception as e:
            print(f"    TabDDPM Ensemble failed: {e}")
    
    # ========================================================================
    # STEP 4: Combine
    # ========================================================================
    if len(X_smote) > 0 and len(X_tabddpm_final) > 0:
        X_combined = np.vstack([X_smote, X_tabddpm_final])
    elif len(X_smote) > 0:
        X_combined = X_smote
    elif len(X_tabddpm_final) > 0:
        X_combined = X_tabddpm_final
    else:
        print("  Both methods failed")
        return None
    
    # Ensure exact count
    if len(X_combined) > n_needed:
        np.random.seed(seed)
        indices = np.random.permutation(len(X_combined))
        X_final = X_combined[indices[:n_needed]]
        
    elif len(X_combined) < n_needed and len(X_combined) > 0:
        additional = n_needed - len(X_combined)
        indices = np.random.choice(len(X_combined), size=additional, replace=True)
        X_final = np.vstack([X_combined, X_combined[indices]])
    elif len(X_combined) == 0:
         print("  Generation failed completely")
         return None
    else:
        X_final = X_combined
    
    # Final statistics
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_minority)
    distances, _ = nn.kneighbors(X_final)
    dcr_final = distances.flatten()
    
    print(f"\n  Final: {len(X_final)} samples")
    print(f"  Composition: {len(X_smote)} SMOTE + {len(X_tabddpm_final)} TabDDPM (DCR-Filtered)")
    print(f"  DCR: mean={dcr_final.mean():.3f}, median={np.median(dcr_final):.3f}")
    
    return X_final
