import tcrkit as tk
import pandas as pd
import numpy as np
import polars as pl
import os
import yaml
import argparse
from typing import Union, List, Dict, Optional, Any, Tuple
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import warnings
from genericpath import exists

# =============================================================================
# CONFIG LOADER
# =============================================================================

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Dictionary with configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"⚠️  Warning: Config file '{config_path}' not found. Using defaults.")
        return None

# =============================================================================
# KEEP ALL YOUR EXISTING FUNCTIONS EXACTLY AS THEY ARE
# =============================================================================

def load_metadata(
    file: str,
    separator: str = ',',
    filter_cell_subset: bool = True,
    column_names: List[str] = ['subject_id', 'visit'],
    disease_label_column: str = 'disease label',
    cell_subset_type: str = 'CD8',
    cell_subset_column: str = 'cell_subset',
) -> pl.DataFrame:
    """
    Load and filter metadata from a CSV file.
    
    Args:
        file: Path to the metadata CSV file
        separator: Column separator in the CSV file
        filter_cell_subset: Whether to filter by cell subset type
        column_names: List of column names to select from the metadata
        disease_label_column: Name of the column containing disease labels
        cell_subset_type: Type of cell subset to filter for (e.g., 'CD8', 'CD4')
        cell_subset_column: Name of the column containing cell subset information
        
    Returns:
        Polars DataFrame with filtered metadata
        
    Raises:
        ValueError: If required parameters are missing or invalid
        
    Examples:
        >>> df = load_metadata('metadata.csv', column_names=['subject_id'])
        >>> print(df.shape)
    """
    if file is None:
        raise ValueError('Metadata file path must be provided')
    
    if len(column_names) == 0:
        raise ValueError('At least one column name must be specified')
    
    if isinstance(column_names, str):
        column_names = [column_names]
    
    if disease_label_column is None:
        raise ValueError('Disease label column must be provided for classification')
    
    # Build column list
    columns_to_select = column_names.copy()
    columns_to_select.append(disease_label_column)
    
    if filter_cell_subset:
        columns_to_select.append(cell_subset_column)
        
        df_sample = (
            pl.scan_csv(
                source=file,
                has_header=True,
                separator=separator
            )
            .select(columns_to_select)
            .filter(pl.col(cell_subset_column) == cell_subset_type)
            .select([col for col in columns_to_select if col != cell_subset_column])
            .collect()
        )
    else:
        df_sample = (
            pl.scan_csv(
                source=file,
                has_header=True,
                separator=separator
            )
            .select(columns_to_select)
            .collect()
        )
    
    return df_sample
def save_simple_dataframe(
    sample_list: List[str],
    metadata_df: pl.DataFrame,
    output_suffix: str,
    output_dir: str = None,
    sample_id_column: str = 'subject_id',
    file_path_column: str = 'aligned_files',
    metadata_columns: List[str] = ['visit', 'child_id'],
    disease_label_column: str = 'seroconversion',
    disease_label_value: str = 'AB'
) -> None:
    """
    Create and save a simple dataframe combining TCR data with metadata.
    
    Args:
        sample_list: List of subject IDs to include
        metadata_df: Metadata DataFrame containing subject info and file paths
        output_suffix: Suffix for output files (e.g., 'fold_train_1')
        output_dir: Base directory for output files
        sample_id_column: Name of column containing sample/subject IDs
        file_path_column: Name of column containing paths to aligned TSV files
        metadata_columns: List of metadata columns to include in output
        disease_label_column: Column to use for creating binary disease label
        disease_label_value: Value that indicates positive disease status (encoded as 1)
    """
    # Validate required columns exist
    required_columns = [sample_id_column, file_path_column, disease_label_column]
    missing_required_columns = [col for col in required_columns if col not in metadata_df.columns]
    if missing_required_columns:
        raise KeyError(f"Required columns missing from metadata: {missing_required_columns}")
    required_columns.extend(metadata_columns)
    missing_columns = [col for col in required_columns if col not in metadata_df.columns]
    if missing_columns:
        warnings.warn(f"Warning : additionnals columns not in the dataframe {missing_columns}")
    
    # Collect file paths for samples
    file_paths = []
    for subject_id in sample_list:
        matches = metadata_df.filter(pl.col(sample_id_column) == subject_id)
        if len(matches) > 0:
            file_paths.append(matches[file_path_column][0])
    
    if len(file_paths) == 0:
        raise ValueError(f"No matching files found for samples: {sample_list}")
    
    # Load TCR data
    df_tcr = load_tsv_files(file_paths)
    
    # Prepare metadata columns to join (include disease_label_column if not in metadata_columns)
    columns_to_join = list(set([sample_id_column] + metadata_columns + [disease_label_column]))
    
    # Join with metadata
    df_combined = (
        df_tcr
        .join(
            metadata_df.select(columns_to_join),
            left_on="sample_id",
            right_on=sample_id_column,
            how="left"
        )
        .with_columns(
            pl.when(pl.col(disease_label_column) == disease_label_value)
            .then(1)
            .otherwise(0)
            .alias('disease_label')
        )
    )
    
    # Create output directory
    dir_path = os.getcwd()
    if output_dir != None :
        output_path = os.path.join(dir_path, output_dir)
        if not exists(output_path):
            os.mkdir(output_path)
    
    else :
        output_path = dir_path
    
    # Save combined dataframe
    output_file = os.path.join(output_path, f'{output_suffix}.csv')
    df_combined.write_csv(file=output_file, separator=',', include_header=True)
    print(f"✓ Saved {len(df_combined)} records to {output_suffix}")


# =============================================================================
# MAIN EXECUTION - Modified to use config.yaml
# =============================================================================

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Create GLIPH2 analysis folders with cross-validation splits'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract parameters from config (with fallback to defaults if config not found)
    if config:
        METADATA_FILE = config['paths']['metadata']
        N_FOLDS = config['n_folds']
        RANDOM_SEED = config.get('random_seed', 42)
        TEST_SIZE = config['cross_validation']['test_size']
        CV_STRATEGY = config['cross_validation']['strategy']
        CV_SHUFFLE = config['cross_validation']['shuffle']
        
        # Metadata parameters
        meta_params = config['metadata']
        SUBJECT_COLUMN = meta_params['subject_column']
        DISEASE_LABEL_COL = meta_params['disease_label_column']
        CELL_SUBSET_TYPE = meta_params['cell_subset_type']
        DISEASE_LABEL_VALUE = 'AB'  # or from config if you add it
        
        # TCRkit parameters (you can add these to config.yaml)
        TCRKIT_CONDITION = config.get('tcrkit', {}).get('condition', 'diabetes')
        TCRKIT_CHAIN = config.get('tcrkit', {}).get('chain', 'TRB')
        TCRKIT_FORMAT = config.get('tcrkit', {}).get('format', 'mixcr')
        TCRKIT_NAME = config.get('tcrkit', {}).get('name', 'T1D_ELICIT_TEST')
        
        print("\n" + "="*70)
        print("Configuration Summary")
        print("="*70)
        print(f"Metadata file: {METADATA_FILE}")
        print(f"Number of folds: {N_FOLDS}")
        print(f"Test size: {TEST_SIZE}")
        print(f"CV strategy: {CV_STRATEGY}")
        print(f"Cell subset: {CELL_SUBSET_TYPE}")
        print(f"Random seed: {RANDOM_SEED}")
        print("="*70 + "\n")
    else:
        # Fallback to hardcoded defaults
        METADATA_FILE = 'clean_metadata.csv'
        N_FOLDS = 5
        RANDOM_SEED = 42
        TEST_SIZE = 0.2
        CV_STRATEGY = 'kfold'
        CV_SHUFFLE = True
        SUBJECT_COLUMN = 'subject_id'
        DISEASE_LABEL_COL = 'seroconversion'
        CELL_SUBSET_TYPE = 'CD8'
        DISEASE_LABEL_VALUE = 'AB'
        TCRKIT_CONDITION = 'diabetes'
        TCRKIT_CHAIN = 'TRB'
        TCRKIT_FORMAT = 'mixcr'
        TCRKIT_NAME = 'T1D_ELICIT_TEST'
    
    # =================================================================
    # STEP 1: Load labels for splitting
    # =================================================================
    print("Loading labels for data splitting...")
    X, y = load_labels(
        file=METADATA_FILE,
        subject_column=SUBJECT_COLUMN,
        disease_label_column=DISEASE_LABEL_COL,
        disease_label=DISEASE_LABEL_VALUE,
        cell_subset_type=CELL_SUBSET_TYPE
    )
    
    # =================================================================
    # STEP 2: Create cross-validation folds
    # =================================================================
    print("\nCreating cross-validation folds...")
    trains, vals, X_test = create_folds(
        X=X, 
        y=y, 
        method=CV_STRATEGY, 
        n_splits=N_FOLDS, 
        seed=RANDOM_SEED,
        test_size=TEST_SIZE
    )
    
    # =================================================================
    # STEP 3: Load the entire metadata
    # =================================================================
    print("\nLoading complete metadata...")
    ref_meta = load_metadata(
        file=METADATA_FILE,
        column_names=['subject_id', 'visit', 'child_id', 'aligned_files'],
        disease_label_column=DISEASE_LABEL_COL,
        filter_cell_subset=True,
        cell_subset_type=CELL_SUBSET_TYPE
    )
    
    # =================================================================
    # STEP 4: Create and save test set
    # =================================================================
    print("\n" + "="*70)
    print("Creating test set")
    print("="*70)
    
    save_simple_dataframe(
        X_test, 
        ref_meta, 
        'test_data'
    )
    
    # =================================================================
    # STEP 5: Create and save training and validation datasets
    # =================================================================
    print("\n" + "="*70)
    print("Creating training and validation sets")
    print("="*70)
    
    for i in range(len(trains)):
        save_simple_dataframe(trains[i], ref_meta, f'fold_train_{i+1}')
        save_simple_dataframe(vals[i], ref_meta, f'fold_val_{i+1}')
    
    # =================================================================
    # STEP 6: Create TCRkit GLIPH2 input files
    # =================================================================
    print("\n" + "="*70)
    print("Creating input GLIPH2 files with TCRkit")
    print("="*70)
    
    # Create the cache path in the working directory
    dir_path = os.getcwd()
    cache_path = os.path.join(dir_path, 'cache')
    
    # Configure the cache
    tk.cache.config(
        cache_dir=cache_path,
        max_size_gb=30,
        max_age_days=140,
        cleanup_on_startup=False
    )
    
    # Create all the tcrkit repertoires
    seronegative = []
    seropositive = []
    
    for i in range(len(trains)):
        print(f"\nProcessing fold {i+1}/{len(trains)}...")
        
        # Load seronegative samples
        seronegative_temp = tk.load_from_metadata(
            METADATA_FILE,
            filepath_column='aligned_files',
            condition=TCRKIT_CONDITION,
            seroconversion='AB-',
            cell_subset=CELL_SUBSET_TYPE,
            chain=TCRKIT_CHAIN,
            format=TCRKIT_FORMAT,
            name=TCRKIT_NAME,
            subject_id=trains[i],
        )
        seronegative.append(seronegative_temp)
        
        # Load seropositive samples
        seropositive_temp = tk.load_from_metadata(
            METADATA_FILE,
            filepath_column='aligned_files',
            condition=TCRKIT_CONDITION,
            seroconversion='AB',
            cell_subset=CELL_SUBSET_TYPE,
            chain=TCRKIT_CHAIN,
            format=TCRKIT_FORMAT,
            name=TCRKIT_NAME,
            subject_id=trains[i],
        )
        seropositive.append(seropositive_temp)
    
    # =================================================================
    # STEP 7: Create GLIPH2 folders with to_gliph()
    # =================================================================
    print("\n" + "="*70)
    print("Creating GLIPH2 reference files and folders")
    print("="*70)
    
    for i in range(len(trains)):
        fold_output_dir = f"./gliph_fold_{i+1}"
        print(f"\nCreating GLIPH2 files for fold {i+1} in {fold_output_dir}...")
        
        # This creates the gliph_fold_X directory AND the GLIPH2 input files!
        gliph_seropositive = seropositive[i].to_gliph(
            output=fold_output_dir,
            condition=TCRKIT_CONDITION,
            reference_mode="other_sample",
            reference_data=seronegative[i],
            data_name="diabetic",
            reference_name="healthy"
        )
        
        print(f"✓ Created {fold_output_dir}")
    
    # =================================================================
    # STEP 8: Save all repertoires to cache
    # =================================================================
    print("\n" + "="*70)
    print("Saving repertoires to cache")
    print("="*70)
    
    for i in range(len(trains)):
        tk.cache.save(f'fold_train_{i+1}', seropositive[i])
        tk.cache.save(f'fold_val_{i+1}', seronegative[i])
    
    print("\n" + "="*70)
    print("✓ All files created successfully!")
    print("="*70)
    print(f"\nCreated:")
    print(f"  - {N_FOLDS} training fold files (fold_train_1.csv to fold_train_{N_FOLDS}.csv)")
    print(f"  - {N_FOLDS} validation fold files (fold_val_1.csv to fold_val_{N_FOLDS}.csv)")
    print(f"  - 1 test file (test_data.csv)")
    print(f"  - {N_FOLDS} GLIPH2 input folders (gliph_fold_1 to gliph_fold_{N_FOLDS})")
    print("="*70 + "\n")