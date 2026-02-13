import pandas as pd
import numpy as np
import yaml
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG LOADER
# =============================================================================

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# =============================================================================
# CONFIGURATION - Now loaded from config.yaml
# =============================================================================

# These will be set from config in main()
N_FOLDS = 5
GLIPH_THRESHOLD_PVALUE = 0.01
MIN_CLUSTER_SIZE = 3
MOTIF_SELECTION_THRESHOLD = 2
MIN_FEATURE_PREVALENCE = 0.01
MAX_FEATURE_PREVALENCE = 0.99
STANDARDIZE_FEATURES = True

def set_config_globals(config: dict):
    """Set global configuration variables from config dict"""
    global N_FOLDS, GLIPH_THRESHOLD_PVALUE, MIN_CLUSTER_SIZE
    global MOTIF_SELECTION_THRESHOLD, MIN_FEATURE_PREVALENCE
    global MAX_FEATURE_PREVALENCE, STANDARDIZE_FEATURES
    
    N_FOLDS = config['n_folds']
    GLIPH_THRESHOLD_PVALUE = config['gliph2']['threshold_pvalue']
    MIN_CLUSTER_SIZE = config['gliph2']['min_cluster_size']
    MOTIF_SELECTION_THRESHOLD = config['classification']['motif_selection_threshold']
    MIN_FEATURE_PREVALENCE = config['classification']['min_feature_prevalence']
    MAX_FEATURE_PREVALENCE = config['classification']['max_feature_prevalence']
    STANDARDIZE_FEATURES = config['classification']['standardize_features']

# =============================================================================
# EXISTING FUNCTIONS (keep as they are)
# =============================================================================

def load_data():
    """Load all data files with error handling."""
    print("Loading data files...")
    
    fold_train_files, fold_val_files, gliph_files = [], [], []
    for fold in range(N_FOLDS) :
        # set the number of train files
        train_file = f'fold_train_{fold+1}.csv'
        fold_train_files.append(train_file)
        
        # set the number of validation files
        val_file = f'fold_val_{fold+1}.csv'
        fold_val_files.append(val_file)
        
        # set the number of train files
        clust_file = f'fold_train_{fold+1}_cluster_clean.csv'
        gliph_files.append(clust_file)
    
    # Load fold data
    try:
        fold_train_data = [pd.read_csv(f) for f in fold_train_files]
        fold_val_data = [pd.read_csv(f) for f in fold_val_files]
        test_data = pd.read_csv('test_data.csv')
        
        # Concatenate all training folds
        train_data = pd.concat(fold_train_data, ignore_index=True)
        
        print(f"✓ Loaded {len(fold_train_files)} training folds")
        print(f"✓ Loaded {len(fold_val_files)} validation folds")
        print(f"✓ Loaded test data:  {len(test_data)} sequences")
        print(f"✓ Total training data: {len(train_data)} sequences")
        
        return fold_train_data, fold_val_data, train_data, test_data, gliph_files
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find file:  {e}")
        print("\nPlease ensure all CSV files are in the current directory:")
        print("  - fold_train_1.csv through fold_train_5.csv")
        print("  - fold_val_1.csv through fold_val_5.csv")
        print("  - test_data.csv")
        print("  - fold_train_1_cluster_clean.csv through fold_train_5_cluster_clean.csv")
        raise

# ============================================================================
# STEP 2: MOTIF MATCHING FUNCTIONS
# ============================================================================
def is_global_motif(motif_string):
    """
    Determine if a motif is global (contains %) or local (pure k-mer).
    """
    return '%' in motif_string


def match_global_motif(cdr3aa, motif_pattern, expected_length):
    """
    Match global motif for a given TCR length.
    Global motif applies to the core region (excluding first 3 and last 3 residues).
    
    Args:
        cdr3aa: Full TCR sequence (e.g., "CASRTQGTEAFF")
        motif_pattern:  Pattern with % as wildcard (e.g., "RTQG%E")
        expected_length: Expected full length of TCR sequence
    
    Returns:  
        bool: True if motif matches
    """
    if len(cdr3aa) != expected_length:
        return False
    
    # Extract core region (excluding first 3 and last 3 residues)
    if len(cdr3aa) < 6:
        return False
    
    core_region = cdr3aa[3:-3]
    
    if len(core_region) != len(motif_pattern):
        return False
    
    # Match pattern with wildcards
    for motif_char, tcr_char in zip(motif_pattern, core_region):
        if motif_char != '%' and motif_char != tcr_char:
            return False
    
    return True


def match_local_motif(cdr3aa, kmer):
    """
    Match local motif (k-mer) anywhere in the TCR sequence.
    
    Args:
        cdr3aa:  Full TCR sequence
        kmer: K-mer pattern to search for
    
    Returns:  
        bool: True if k-mer is found in sequence
    """
    return kmer in cdr3aa


def tcr_matches_motif(cdr3aa, motif_info):
    """
    Check if a TCR sequence matches a motif (dispatches to appropriate matcher).
    
    Args:
        cdr3aa: TCR sequence string
        motif_info: Dictionary containing motif information
    
    Returns:
        bool: True if TCR matches the motif
    """
    if motif_info['type'] == 'global':
        return match_global_motif(
            cdr3aa,
            motif_info['pattern'],
            motif_info['tcr_length']
        )
    else:
        return match_local_motif(cdr3aa, motif_info['pattern'])


# ============================================================================
# STEP 3: EXTRACT MOTIFS FROM GLIPH2 FILES
# ============================================================================
def parse_gliph2_file(filepath, pvalue_threshold=0.01, min_cluster_size=2):
    """
    Parse GLIPH2 output and extract significant motifs.
    
    GLIPH2 CSV format with header:
    - pattern: Motif pattern (with % as wildcard for global motifs)
    - Fisher_score: P-value for motif significance
    - TcRb:  TCR CDR3 beta sequence
    - number_unique_cdr3: Number of unique TCR sequences in cluster
    
    Args:
        filepath: Path to GLIPH2 output file
        pvalue_threshold:  Maximum p-value for significance (default: 0.01)
        min_cluster_size:  Minimum number of unique TCRs per cluster (default: 2)
    
    Returns:
        dict: Dictionary of motif_id -> motif_info
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:  
        print(f"❌ Error reading {filepath}: {e}")
        return {}
    
    print(f"Loaded {len(df)} rows from {filepath}")
    
    # Check required columns
    required_cols = ['pattern', 'Fisher_score', 'TcRb', 'number_unique_cdr3']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return {}
    
    # Extract relevant columns
    MOTIF_PATTERN_COL = 'pattern'
    PVALUE_COL = 'Fisher_score'
    CDR3AA_COL = 'TcRb'
    UNIQUE_TCR_COL = 'number_unique_cdr3'
    
    # Filter by p-value
    df_filtered = df[df[PVALUE_COL] < pvalue_threshold].copy()
    print(f"After p-value filter: {len(df_filtered)} rows (Fisher_score < {pvalue_threshold})")
    
    if len(df_filtered) == 0:
        print(f"⚠ WARNING: No motifs pass p-value threshold")
        return {}
    
    # Extract motifs
    motifs = {}
    skipped_small_clusters = 0
    
    for motif_pattern in df_filtered[MOTIF_PATTERN_COL].unique():
        # Skip NaN patterns
        if pd.isna(motif_pattern):
            continue
            
        motif_rows = df_filtered[df_filtered[MOTIF_PATTERN_COL] == motif_pattern]
        
        # Get unique TCR sequences for this motif
        cdr3aas = motif_rows[CDR3AA_COL].dropna().unique()
        
        # Get number of unique CDR3 sequences from the data
        # Use the maximum value if there are multiple rows (should be the same)
        number_sequences = motif_rows[UNIQUE_TCR_COL].max()
        
        # Check cluster size
        if number_sequences < min_cluster_size:  
            skipped_small_clusters += 1
            continue
        
        # Get TCR length from the first valid sequence
        valid_tcrs = [tcr for tcr in cdr3aas if isinstance(tcr, str) and len(tcr) > 0]
        if not valid_tcrs:
            continue
        
        tcr_length = len(valid_tcrs[0])
        
        # Get best (minimum) p-value for this motif
        pvalue = motif_rows[PVALUE_COL].min()
        
        # Determine motif type and create ID
        if is_global_motif(motif_pattern):
            motif_id = f"GLOBAL_{motif_pattern}_L{int(tcr_length)}"
            motif_type = 'global'
        else:
            motif_id = f"LOCAL_{motif_pattern}"
            motif_type = 'local'
        
        # Store motif information
        motifs[motif_id] = {
            'pattern': motif_pattern,
            'type': motif_type,
            'length': len(motif_pattern),
            'tcr_length': int(tcr_length),
            'tcrs': set(valid_tcrs),
            'pvalue': pvalue,
            'cluster_size': int(number_sequences)
        }
    
    if skipped_small_clusters > 0:
        print(f"Skipped {skipped_small_clusters} motifs with cluster size < {min_cluster_size}")
    
    return motifs


def extract_motifs_from_all_folds(gliph_files):
    """
    Extract motifs from all GLIPH2 fold files.
    
    Args:
        gliph_files:  List of paths to GLIPH2 output files
    
    Returns:  
        list: List of motif dictionaries (one per fold)
    """
    fold_motifs = []
    
    print("\n" + "="*70)
    print("EXTRACTING MOTIFS FROM GLIPH2 FILES")
    print("="*70)
    
    for i, filepath in enumerate(gliph_files):
        print(f"\n[Fold {i+1}/{len(gliph_files)}]")
        motifs = parse_gliph2_file(filepath, GLIPH_THRESHOLD_PVALUE, MIN_CLUSTER_SIZE)
        fold_motifs.append(motifs)
        
        # Summary statistics
        n_global = sum(1 for m in motifs.values() if m['type'] == 'global')
        n_local = sum(1 for m in motifs.values() if m['type'] == 'local')
        total_tcrs = sum(len(m['tcrs']) for m in motifs.values())
        
        print(f"✓ Found {len(motifs)} motifs ({n_global} global, {n_local} local)")
        print(f"✓ Total TCR sequences in clusters: {total_tcrs}")
    
    # Overall summary
    total_unique_motifs = len(set().union(*[set(m.keys()) for m in fold_motifs]))
    print(f"\n{'='*70}")
    print(f"Total unique motifs across all folds: {total_unique_motifs}")
    print(f"{'='*70}")
    
    return fold_motifs


# ============================================================================
# STEP 4: CREATE FEATURE MATRIX
# ============================================================================
def create_motif_features(data, motifs_dict, verbose=True):
    """
    Create frequency-based feature matrix:  proportion of patient's TCRs matching each motif.
    
    Args:
        data: DataFrame with 'cdr3aa' and 'sample_id' columns
        motifs_dict: Dictionary of motifs from GLIPH2
        verbose:  Print progress information
    
    Returns:
        DataFrame with sample_id as index and motifs as columns (frequency values)
    """
    # Get unique patients
    patients = data['sample_id'].unique()
    
    if verbose:
        print(f"Creating frequency features for {len(patients)} patients and {len(motifs_dict)} motifs...")
    
    # Initialize feature matrix
    feature_matrix = pd.DataFrame(index=patients)
    
    # Group TCR sequences by patient for efficiency
    patient_tcrs = data.groupby('sample_id')['cdr3aa'].apply(list).to_dict()
    
    # For each motif, compute frequency
    for motif_idx, (motif_id, motif_info) in enumerate(motifs_dict.items()):
        if verbose and (motif_idx + 1) % 100 == 0:
            print(f"Progress: {motif_idx + 1}/{len(motifs_dict)} motifs processed")
        
        patient_motif_frequencies = {}
        
        for patient in patients:
            tcrs = patient_tcrs.get(patient, [])
            
            if len(tcrs) == 0:
                patient_motif_frequencies[patient] = 0.0
                continue
            
            # Count matching TCRs
            matching_count = sum(
                1 for tcr in tcrs
                if tcr_matches_motif(str(tcr), motif_info)
            )
            
            # Compute frequency (proportion)
            frequency = matching_count / len(tcrs)
            patient_motif_frequencies[patient] = frequency
        
        feature_matrix[motif_id] = pd.Series(patient_motif_frequencies)
    
    if verbose:  
        print(f"✓ Feature matrix created:  {feature_matrix.shape}")
        print(f"Feature statistics:")
        print(f"  Mean: {feature_matrix.values.mean():.4f}")
        print(f"  Median: {np.median(feature_matrix.values):.4f}")
        print(f"  Max: {feature_matrix.values.max():.4f}")
        print(f"  % non-zero: {(feature_matrix > 0).values.mean() * 100:.2f}%")
    
    return feature_matrix


def standardize_features(X_train, X_val=None, X_test=None):
    """
    Standardize features to zero mean and unit variance.
    Fit on training data, transform all sets.
    
    Args:
        X_train: Training feature matrix
        X_val: Validation feature matrix (optional)
        X_test: Test feature matrix (optional)
    
    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
        Returns None for optional sets if not provided
    """
    scaler = StandardScaler()
    
    # Fit on training data only
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )
    
    # Transform validation and test with same scaler
    X_val_scaled = None
    if X_val is not None:
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            index=X_val.index,
            columns=X_val.columns
        )
    
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            index=X_test.index,
            columns=X_test.columns
        )
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def filter_features_by_prevalence(X, min_prev=0.01, max_prev=0.99):
    """
    Remove features that are too rare or too common.
    
    Args:
        X: Feature matrix
        min_prev: Minimum prevalence (fraction of samples with feature > 0)
        max_prev: Maximum prevalence
    
    Returns:
        Filtered feature matrix and list of removed features
    """
    prevalence = (X > 0).mean(axis=0)
    
    keep_features = X.columns[(prevalence >= min_prev) & (prevalence <= max_prev)]
    removed_features = X.columns[(prevalence < min_prev) | (prevalence > max_prev)]
    
    return X[keep_features], list(removed_features)


def select_motifs_across_folds(fold_train_data,
                               fold_val_data,
                               fold_motifs,
                               n_folds=5,motif_threshold=None,):
    """
    Select motifs based on cross-validation performance.
    Includes feature standardization. 
    """
    if motif_threshold is None:
        motif_threshold = MOTIF_SELECTION_THRESHOLD
    # Track motif performance across folds
    motif_coefficients = defaultdict(list)
    motif_appears_in_folds = defaultdict(int)
    
    print("\n" + "="*70)
    print("CROSS-VALIDATION:  MOTIF SELECTION")
    print("="*70)
    
    for fold_idx in range(n_folds):
        print(f"\n{'─'*70}")
        print(f"Processing Fold {fold_idx + 1}/{n_folds}")
        print(f"{'─'*70}")
        
        # Get this fold's data
        train_fold = fold_train_data[fold_idx]
        val_fold = fold_val_data[fold_idx]
        
        # Get patient counts
        train_patients = train_fold['sample_id'].unique()
        val_patients = val_fold['sample_id'].unique()
        
        print(f"Patients - Train: {len(train_patients)}, Validation: {len(val_patients)}")
        
        # Get motifs for this fold
        fold_motif_dict = fold_motifs[fold_idx]
        print(f"  Motifs available: {len(fold_motif_dict)}")
        
        if len(fold_motif_dict) == 0:
            print("⚠ WARNING: No motifs found for this fold, skipping...")
            continue
        
        # Create frequency features
        print(f"Creating frequency feature matrices...")
        X_train_fold = create_motif_features(train_fold, fold_motif_dict, verbose=False)
        X_val_fold = create_motif_features(val_fold, fold_motif_dict, verbose=False)
        
        # Get labels
        y_train_fold = train_fold.groupby('sample_id')['disease_label'].first().loc[X_train_fold.index].values
        y_val_fold = val_fold.groupby('sample_id')['disease_label'].first().loc[X_val_fold.index].values
        
        print(f"Class distribution (train): {dict(Counter(y_train_fold))}")
        print(f"Class distribution (val): {dict(Counter(y_val_fold))}")
        
        # Remove features with no variation or extreme prevalence
        motifs_with_variation = []
        for col in X_train_fold.columns:
            prevalence = (X_train_fold[col] > 0).mean()  # Proportion of patients with motif
            if X_train_fold[col].nunique() >= 2 and MIN_FEATURE_PREVALENCE <= prevalence <= MAX_FEATURE_PREVALENCE:
                motifs_with_variation.append(col)
        
        X_train_fold = X_train_fold[motifs_with_variation]
        X_val_fold = X_val_fold[motifs_with_variation]
        
        print(f"Features after filtering: {len(motifs_with_variation)}")
        
        if len(motifs_with_variation) == 0:
            print("⚠ WARNING: No motifs passed filtering, skipping fold...")
            continue
        
        # Standardize features if enabled
        if STANDARDIZE_FEATURES:
            print(f"Standardizing features...")
            X_train_fold, X_val_fold, _, _ = standardize_features(X_train_fold, X_val_fold)
        
        # Get number of CPUs from SLURM (if available)
        N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', -1))

        print(f"Using {N_JOBS if N_JOBS > 0 else 'all available'} CPUs for parallel processing")

        # Train logistic regression
        print(f"Training logistic regression...")
        lr = LogisticRegression(
            solver='saga',
            l1_ratio=0.5,
            C=1.0,
            class_weight='balanced',
            max_iter=5000,
            random_state=42,
            n_jobs=N_JOBS,  # Uses SLURM allocation
            verbose=0
        )
        
        try:
            lr.fit(X_train_fold, y_train_fold)
            
            # Evaluate on validation set
            y_val_proba = lr.predict_proba(X_val_fold)[:, 1]
            val_auc = roc_auc_score(y_val_fold, y_val_proba)
            
            y_train_proba = lr.predict_proba(X_train_fold)[:, 1]
            train_auc = roc_auc_score(y_train_fold, y_train_proba)
            
            print(f"✓ Training AUC: {train_auc:.4f}")
            print(f"✓ Validation AUC: {val_auc:.4f}")
            
            # Store coefficient for each motif
            for motif_id, coef in zip(X_train_fold.columns, lr.coef_[0]):
                motif_appears_in_folds[motif_id] += 1
                motif_coefficients[motif_id].append(coef)
            
            print(f"✓ Fold {fold_idx + 1} complete")
            
        except Exception as e:
            print(f"❌ ERROR training model: {e}")
            continue
    
    # Select best motifs based on coefficient consistency
    print(f"\n{'='*70}")
    print("SELECTING BEST MOTIFS ACROSS FOLDS")
    print(f"{'='*70}")
    
    selected_motifs = []
    motif_scores = {}
    
    for motif_id in motif_appears_in_folds.keys():
        n_folds_appeared = motif_appears_in_folds[motif_id]
        
        # Get performance metrics
        coefs = motif_coefficients.get(motif_id, [])
        
        if len(coefs) == 0:
            continue
        
        mean_coef = np.mean(coefs)
        std_coef = np.std(coefs)
        abs_mean_coef = np.abs(mean_coef)
        
        # Check coefficient consistency (same sign across folds)
        coef_signs = [np.sign(c) for c in coefs if c != 0]
        if len(coef_signs) > 0:
            sign_consistency = sum(1 for s in coef_signs if s == np.sign(mean_coef)) / len(coef_signs)
        else:
            sign_consistency = 0
        
        # Selection criteria
        is_selected = ( 
            n_folds_appeared >= MOTIF_SELECTION_THRESHOLD and
            sign_consistency >= 0.7
        )
        
        if is_selected: 
            selected_motifs.append(motif_id)
            motif_scores[motif_id] = {
                'mean_coef': mean_coef,
                'std_coef':  std_coef,
                'abs_mean_coef': abs_mean_coef,
                'sign_consistency': sign_consistency,
                'n_folds_appeared': n_folds_appeared,
                'all_coefs': coefs
            }
    
    # Sort by coefficient magnitude and consistency
    selected_motifs = sorted(
        selected_motifs,
        key=lambda m: (motif_scores[m]['sign_consistency'], motif_scores[m]['abs_mean_coef']),
        reverse=True
    )
    
    print(f"\n✓ Selected {len(selected_motifs)} motifs")
    
    # Display top motifs
    if len(selected_motifs) > 0:
        print(f"\nTop 20 selected motifs:")
        for i, motif_id in enumerate(selected_motifs[:20], 1):
            score = motif_scores[motif_id]
            print(f"  {i:2d}.{motif_id:45s} | "
                  f"Coef: {score['mean_coef']:+.3f}±{score['std_coef']:.3f} | "
                  f"Consistency: {score['sign_consistency']:.1%} | "
                  f"{score['n_folds_appeared']}/{n_folds} folds")
    
    return selected_motifs, motif_scores


# ============================================================================
# STEP 6: TRAIN FINAL MODEL ON FULL TRAINING SET
# ============================================================================
def merge_motifs_from_folds(fold_motifs, selected_motifs):
    """
    Merge motif definitions from all folds, keeping the best version of each motif.
    
    Args:
        fold_motifs: List of motif dictionaries from each fold
        selected_motifs: List of selected motif IDs
    
    Returns:  
        dict:  Merged motif dictionary with best versions
    """
    merged_motifs = {}
    
    for motif_id in selected_motifs:
        # Find this motif across all folds
        motif_versions = []
        for fold_motif_dict in fold_motifs:  
            if motif_id in fold_motif_dict: 
                motif_versions.append(fold_motif_dict[motif_id])
        
        if motif_versions:
            # Keep the version with the best (lowest) p-value
            best_version = min(motif_versions, key=lambda m: m['pvalue'])
            merged_motifs[motif_id] = best_version
    
    return merged_motifs

def train_final_model(train_data, test_data, selected_motifs, fold_motifs):
    """
    Train final regularized logistic regression on selected motifs.
    Uses FULL training set with frequency features and standardization.
    """
    print(f"\n{'='*70}")
    print("TRAINING FINAL MODEL ON FULL TRAINING SET")
    print(f"{'='*70}")
    
    # Merge motifs from all folds
    final_motifs = merge_motifs_from_folds(fold_motifs, selected_motifs)
    
    print(f"\nUsing {len(final_motifs)} selected motifs")
    
    # Create frequency feature matrices
    print(f"\nCreating frequency feature matrices...")
    print(f"  [1/2] Processing training data...")
    X_train = create_motif_features(train_data, final_motifs, verbose=True)
    print(f"  [2/2] Processing test data...")
    X_test = create_motif_features(test_data, final_motifs, verbose=True)
    
    # Get labels
    y_train = train_data.groupby('sample_id')['disease_label'].first().loc[X_train.index].values
    y_test = test_data.groupby('sample_id')['disease_label'].first().loc[X_test.index].values
    
    print(f"\nDataset summary:")
    print(f"  Training: {X_train.shape[0]} patients, {X_train.shape[1]} motif features")
    print(f"  Test: {X_test.shape[0]} patients")
    print(f"  Class distribution (train): {dict(Counter(y_train))}")
    print(f"  Class distribution (test): {dict(Counter(y_test))}")
    
    # Feature type breakdown
    n_global = sum(1 for m in final_motifs.values() if m['type'] == 'global')
    n_local = sum(1 for m in final_motifs.values() if m['type'] == 'local')
    print(f"  Feature types: {n_global} global motifs, {n_local} local motifs")
    
    # Standardize features if enabled
    if STANDARDIZE_FEATURES:  
        print(f"\nStandardizing features...")
        X_train, _, X_test, scaler = standardize_features(X_train, X_test=X_test)
        print(f"✓ Features standardized (mean=0, std=1)")
    else:
        scaler = None
    
    # Train final model
    # Get number of CPUs from SLURM (if available)
    N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', -1))

    print(f"\nUsing {N_JOBS if N_JOBS > 0 else 'all available'} CPUs for parallel processing")

    # Train logistic regression
    print(f"Training logistic regression...")
    final_model = LogisticRegression( # Changer d'après le hyperparameter tuning
        solver='saga',
        l1_ratio=0.0,
        C=1.0,
        class_weight='balanced',
        max_iter=5000,
        random_state=42,
        n_jobs=N_JOBS,  # Uses SLURM allocation
        verbose=0
    )
    
    final_model.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = final_model.predict(X_train)
    y_train_proba = final_model.predict_proba(X_train)[:, 1]
    
    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    # Print results
    print(f"\n{'='*70}")
    print("FINAL RESULTS ON HELD-OUT TEST SET")
    print(f"{'='*70}")
    print(f"\nAUC Scores:")
    print(f"  Training: {train_auc:.4f}")
    print(f"  Test:      {test_auc:.4f}")
    
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred, digits=4))
    
    print("\nTest Set Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'motif': X_train.columns,
        'coefficient': final_model.coef_[0],
        'motif_type': [final_motifs[m]['type'] for m in X_train.columns],
        'pattern': [final_motifs[m]['pattern'] for m in X_train.columns],
        'cluster_size': [final_motifs[m]['cluster_size'] for m in X_train.columns],
        'pvalue': [final_motifs[m]['pvalue'] for m in X_train.columns]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\nTop 15 Most Important Motifs (by |coefficient|):")
    print(feature_importance.head(15).to_string(index=False))
    
    # Count non-zero coefficients
    n_nonzero = np.sum(final_model.coef_[0] != 0)
    print(f"\nFeatures with non-zero coefficients: {n_nonzero}/{len(final_model.coef_[0])}")
    
    return final_model, X_train, X_test, y_train, y_test, feature_importance, final_motifs, scaler

# ============================================================================
# STEP 7: VISUALIZATION
# ============================================================================
def plot_results(y_test, y_test_proba, feature_importance, output_file='tcr_classification_results.png'):
    """
    Create comprehensive visualizations of model performance.
    
    Args:
        y_test: True test labels
        y_test_proba: Predicted probabilities
        feature_importance: DataFrame with feature importance
        output_file: Path to save figure
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    
    print(f"\nCreating visualizations...")
    
    fig = plt.figure(figsize=(18, 6))
    
    # 1. Enhanced ROC Curve
    ax1 = plt.subplot(1, 3, 1)
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot with filled area
    ax1.plot(fpr, tpr, color='#2E86AB', lw=3, label=f'AUC = {roc_auc:.3f}')
    ax1.fill_between(fpr, tpr, alpha=0.3, color='#2E86AB')
    ax1.plot([0, 1], [0, 1], color='#A23B72', lw=2, linestyle='--', 
             label='Random Classifier', alpha=0.7)
    
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)
    ax1.grid(alpha=0.4, linestyle='--', linewidth=0.7)
    ax1.set_facecolor('#F8F9FA')
    
    # 2. Enhanced Precision-Recall Curve
    ax2 = plt.subplot(1, 3, 2)
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    avg_precision = average_precision_score(y_test, y_test_proba)
    
    ax2.plot(recall, precision, color='#F18F01', lw=3, label=f'AP = {avg_precision:.3f}')
    ax2.fill_between(recall, precision, alpha=0.3, color='#F18F01')
    
    # Baseline (proportion of positive class)
    baseline = np.sum(y_test) / len(y_test)
    ax2.axhline(y=baseline, color='#A23B72', linestyle='--', lw=2, 
                label=f'Baseline = {baseline:.3f}', alpha=0.7)
    
    ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc="best", fontsize=11, frameon=True, shadow=True)
    ax2.grid(alpha=0.4, linestyle='--', linewidth=0.7)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.set_facecolor('#F8F9FA')
    
    # 3. Confusion Matrix Heatmap
    ax3 = plt.subplot(1, 3, 3)
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with both count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    # Create heatmap
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=True,
                square=True, linewidths=2, linecolor='white',
                cbar_kws={'label': 'Count', 'shrink': 0.8},
                vmin=0, vmax=cm.max(),
                annot_kws={'fontsize': 13, 'fontweight': 'bold'})
    
    ax3.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax3.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xticklabels(['Healthy (0)', 'Disease (1)'], fontsize=11)
    ax3.set_yticklabels(['Healthy (0)', 'Disease (1)'], fontsize=11, rotation=0)
    
    # Add text for metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics_text = f'Accuracy: {accuracy:.3f}\nSensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}'
    ax3.text(1.02, 0.5, metrics_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Plot saved as '{output_file}'")
    plt.close()


# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================
def save_results(final_model, final_motifs, selected_motifs, motif_scores, feature_importance, scaler=None):
    """
    Save all model artifacts and results.
    Includes scaler for feature standardization.
    """
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    # Save model
    joblib.dump(final_model, 'final_tcr_model.pkl')
    print("✓ Model saved as 'final_tcr_model.pkl'")
    
    # Save scaler if used
    if scaler is not None:
        joblib.dump(scaler, 'feature_scaler.pkl')
        print("✓ Feature scaler saved as 'feature_scaler.pkl'")
    
    # Save motif information
    joblib.dump(final_motifs, 'final_motifs_info.pkl')
    print("✓ Motif information saved as 'final_motifs_info.pkl'")
    
    # Save selected motifs with detailed statistics
    motif_details = []
    for motif_id in selected_motifs:  
        if motif_id in final_motifs:
            info = final_motifs[motif_id]
            score = motif_scores[motif_id]
            motif_details.append({
                'motif_id': motif_id,
                'pattern': info['pattern'],
                'type': info['type'],
                'tcr_length': info.get('tcr_length', 'N/A'),
                'cluster_size': info['cluster_size'],
                'pvalue': info['pvalue'],
                'mean_coef': score['mean_coef'],
                'std_coef': score['std_coef'],
                'sign_consistency': score['sign_consistency'],
                'n_folds_appeared': score['n_folds_appeared']
            })
    
    pd.DataFrame(motif_details).to_csv('selected_motifs.csv', index=False)
    print("✓ Selected motifs saved as 'selected_motifs.csv'")
    
    # Save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)
    print("✓ Feature importance saved as 'feature_importance.csv'")
    
    print(f"\n{'='*70}")
    print("All results saved successfully!")
    print(f"{'='*70}")


# =============================================================================
# MAIN EXECUTION with CONFIG
# =============================================================================

def main():
    """Main function that uses config.yaml parameters"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='TCR motif classification')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    set_config_globals(config)

    """Main execution pipeline."""
    print("="*70)
    print("TCR Motif Classification")
    print("="*70)
    print("\nThis pipeline performs the following steps:")
    print("  1. Extract motifs from GLIPH2 files")
    print("  2. Select robust motifs via cross-validation")
    print("  3. Train final model on full training set")
    print("  4. Evaluate on held-out test set")
    print("  5. Save results and visualizations")
    print("="*70)
    print(f"\nConfiguration loaded from: {args.config}")
    print(f"Number of folds: {N_FOLDS}")
    print(f"GLIPH2 p-value threshold: {GLIPH_THRESHOLD_PVALUE}")
    print(f"Motif selection threshold: {MOTIF_SELECTION_THRESHOLD}")
    print(f"Standardize features: {STANDARDIZE_FEATURES}")
    print("="*60 + "\n")
    
    try:
        # Step 0: Load data
        print("\n[0/5] Loading data...")
        fold_train_data, fold_val_data, train_data, test_data, gliph_files = load_data()
        
        # Step 1: Extract motifs from GLIPH2 files
        print("\n[1/5] Extracting motifs from GLIPH2 files...")
        fold_motifs = extract_motifs_from_all_folds(gliph_files)
        
        # Step 2: Select motifs based on cross-validation performance
        print("\n[2/5] Selecting motifs across validation folds...")
        selected_motifs, motif_scores = select_motifs_across_folds(
            fold_train_data, fold_val_data, fold_motifs, n_folds=N_FOLDS
        )
        
        # Handle case where no motifs pass initial criteria
        if len(selected_motifs) == 0:
            print("\n⚠ WARNING: No motifs passed selection criteria!")
            print("Relaxing thresholds and retrying...")
            
            selected_motifs, motif_scores = select_motifs_across_folds(
                fold_train_data, fold_val_data, fold_motifs, n_folds=N_FOLDS, motif_threshold=1
            )
        
        if len(selected_motifs) == 0:
            print("\n❌ ERROR: Still no motifs selected after relaxing criteria.")
            print("Please check your GLIPH2 output and data quality.")
            return
        
        # Step 3: Train final model on FULL training set
        print("\n[3/5] Training final model on full training set...")
        final_model, X_train, X_test, y_train, y_test, feature_importance, final_motifs, scaler = train_final_model(
            train_data, test_data, selected_motifs, fold_motifs
        )
        
        # Step 4: Visualize results
        print("\n[4/5] Creating visualizations...")
        y_test_proba = final_model.predict_proba(X_test)[:, 1]
        plot_results(y_test, y_test_proba, feature_importance)
        
        # Step 5: Save all results
        print("\n[5/5] Saving results...")
        save_results(final_model, final_motifs, selected_motifs, motif_scores, feature_importance, scaler)
        
        # Final summary
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\nFinal Model Performance:")
        print(f"  Test AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
        print(f"  Selected Motifs: {len(selected_motifs)}")
        print(f"  Non-zero Coefficients: {np.sum(final_model.coef_[0] != 0)}")
        print("\nOutput files:")
        print("  - final_tcr_model.pkl")
        if scaler is not None:
            print("  - feature_scaler.pkl")
        print("  - final_motifs_info.pkl")
        print("  - selected_motifs.csv")
        print("  - feature_importance.csv")
        print("  - tcr_classification_results.png")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED")
        print(f"Error:  {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()