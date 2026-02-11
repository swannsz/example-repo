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
    # ... (keep your existing implementation)
    pass

# ... (all your other existing functions)

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
            
            global MOTIF_SELECTION_THRESHOLD, PERFORMANCE_THRESHOLD_AUC
            MOTIF_SELECTION_THRESHOLD = 1
            PERFORMANCE_THRESHOLD_AUC = 0.55
            
            selected_motifs, motif_scores = select_motifs_across_folds(
                fold_train_data, fold_val_data, fold_motifs, n_folds=N_FOLDS
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