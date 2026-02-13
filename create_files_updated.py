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

def load_labels(
    file: str,
    separator: str = ',',
    filter_cell_subset: bool = True,
    subject_column: str = 'subject_id',
    disease_label_column: str = 'seroconversion',
    cell_subset_type: str = 'CD8',
    cell_subset_column: str = 'cell_subset',
    disease_label: Optional[str] = None
) -> Tuple[pd.Series, pd.Series]:
    """
    Load subject IDs and disease labels for train/test splitting.
    
    Args:
        file: Path to the metadata CSV file
        separator: Column separator in the CSV file
        filter_cell_subset: Whether to filter by cell subset type
        subject_column: Name of the column containing subject IDs
        disease_label_column: Name of the column containing disease labels
        cell_subset_type: Type of cell subset to filter for
        cell_subset_column: Name of the column containing cell subset information
        disease_label: Specific disease label to encode as 1 (others as 0). 
                      If None, keeps original labels.
        
    Returns:
        Tuple of (subject_ids, labels) as pandas Series
        
    Raises:
        ValueError: If required parameters are missing
        
    Examples:
        >>> X, y = load_labels('metadata.csv', disease_label='AB')
        >>> print(f"Loaded {len(X)} samples")
    """
    if file is None:
        raise ValueError('Metadata file path must be provided')
    
    if subject_column is None or len(subject_column) == 0:
        raise ValueError('Subject ID column must be provided for data splitting')
    
    if disease_label_column is None:
        raise ValueError('Disease label column must be provided for classification')
    
    # Load metadata
    df_sample = load_metadata(
        file=file,
        separator=separator,
        filter_cell_subset=filter_cell_subset,
        column_names=[subject_column],
        disease_label_column=disease_label_column,
        cell_subset_type=cell_subset_type,
        cell_subset_column=cell_subset_column
    )
    
    columns_to_select = [subject_column, disease_label_column]
    
    # Binary encoding if specific disease label provided
    if isinstance(disease_label, str):
        df_sample_for_splitting = (
            df_sample
            .select(columns_to_select)
            .with_columns(
                pl.when(pl.col(disease_label_column) == disease_label)
                .then(1)
                .otherwise(0)
                .alias(disease_label_column)
            )
        )
    else:
        df_sample_for_splitting = df_sample.select(columns_to_select)
    
    df_sample_pd = df_sample_for_splitting.to_pandas()
    
    return df_sample_pd[subject_column], df_sample_pd[disease_label_column]

def create_folds(
    X: pd.Series,
    y: pd.Series,
    method: str = 'kfold',
    n_splits: int = 5,
    seed: int = 42,
    test_size: float = 0.2
) -> Tuple[List[List], List[List], List]:
    """
    Create train/validation/test splits with nested cross-validation.
    
    First splits data into train+validation and test sets.
    Then creates k-fold splits within the train set for cross-validation.
    
    Args:
        X: Subject IDs
        y: Disease labels
        method: Splitting method ('kfold' or 'stratified')
        n_splits: Number of folds for k-fold cross-validation within training data
        seed: Random seed for reproducibility
        test_size: Proportion of full data reserved for final test set
        
    Returns:
        Tuple of (train_folds, val_folds, test_set):
        - train_folds: List of training subject ID lists (one per fold)
        - val_folds: List of validation subject ID lists (one per fold)
        - test_set: List of test subject IDs (held out from CV)
    
    Warning :
        stratified method is not a good way of proceeding if the dataset is to small. It will indeed
        create only one possible splitting
    """
    # First, split into train+val and test
    if method == 'kfold' :
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=seed, 
        )
    elif method == 'stratified':
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=seed, 
            stratify=y
        )
    
    print("Train class distribution:")
    print (f'0 : {len(y_trainval) - y_trainval.sum()}\n1 : {y_trainval.sum()}')
    print("\nTest class distribution:")
    print (f'0 : {len(y_test) - y_test.sum()}\n1 : {y_test.sum()}')
        
    train_folds = []
    val_folds = []
    
    # Then create k-fold splits within the train+val set
    if method == 'kfold':
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splitter = kf.split(X_trainval)
    elif method == 'stratified':
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splitter = skf.split(X_trainval, y_trainval)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kfold' or 'stratified'")
    
    # Create folds within training data
    i = 0
    fold_data = []

    for train_idx, val_idx in splitter:
        i += 1
        train_folds.append(X_trainval.iloc[train_idx].tolist())
        val_folds.append(X_trainval.iloc[val_idx].tolist())
        
        # Collect fold statistics
        fold_data.append({
            'Fold': i,
            'Split': 'Train',
            'Class 0': len(y_trainval.iloc[train_idx]) - y_trainval.iloc[train_idx].sum(),
            'Class 1': y_trainval.iloc[train_idx].sum()
        })
        fold_data.append({
            'Fold': i,
            'Split': 'Validation',
            'Class 0': len(y_trainval.iloc[val_idx]) - y_trainval.iloc[val_idx].sum(),
            'Class 1': y_trainval.iloc[val_idx].sum()
        })

    # Create and display single comprehensive table
    fold_df = pd.DataFrame(fold_data)
    print("\nClass distribution across all folds:")
    print(fold_df.to_string(index=False))
    
    return train_folds, val_folds, X_test.tolist()

def load_tsv_files(
    file_paths: Union[str, List[str]],
    cdr3_column: str = 'aaSeqCDR3',
    count_column: str = 'cloneCount',
    v_gene_column: str = 'allVHitsWithScore',
    j_gene_column: str = 'allJHitsWithScore',
    min_cdr3_length: int = 6,
    max_cdr3_length: int = 22,
) -> pl.DataFrame:
    """
    Load and process TCR repertoire data from TSV files.
        
    Args:
        file_paths: Path to single TSV file or list of paths to multiple TSV files
        cdr3_column: Name of column containing CDR3 amino acid sequence
        count_column: Name of column containing clone counts
        v_gene_column: Name of column containing V gene information
        j_gene_column: Name of column containing J gene information
        min_cdr3_length: Minimum CDR3 length to include (inclusive, default: 6)
        max_cdr3_length: Maximum CDR3 length to include (exclusive, default: 22)
        
    Returns:
        Polars DataFrame with processed TCR sequences containing:
        - cdr3aa: CDR3 amino acid sequence (always starts with C, ends with F)
        - count: Clone count
        - v_gene: V gene (optionally without allele)
        - j_gene: J gene (optionally without allele)
        - sample_id: Sample identifier extracted from file path
        - cdr3Length: Length of CDR3 sequence
        - clone: Unique clone identifier (v_gene_cdr3aa_j_gene)
    """
    
    # Convert single path to list for consistent handling
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    
    # Start building the lazy query
    query = (
        pl.scan_csv(
            file_paths,
            separator="\t",
            has_header=True,
            include_file_paths="file_path"
        )
        .select([
            cdr3_column,
            count_column,
            v_gene_column,
            j_gene_column,
            "file_path"
        ])
    )
    
    #remove TRAJ and at the same occasion all alpha chains
    query = query.filter(~pl.col(j_gene_column).str.contains('A'))
    
    # Extract sample ID from file path
    query = query.with_columns(
        pl.col("file_path")
        .str.extract(r"([^/]+)-prod\.tsv$", 1)
        .alias("sample_id")
    )
    
    # Rename columns to standard names
    query = query.rename({
        cdr3_column: "cdr3aa",
        count_column: "count",
        v_gene_column: "v_gene",
        j_gene_column: "j_gene",
    })
    
    # Remove allele information
    query = query.with_columns([
        pl.col("v_gene").str.replace(r"\*.*", ""),
        pl.col("j_gene").str.replace(r"\*.*", ""),
    ])
    
    # Add CDR3 length and create clone identifier
    query = (
        query
        .with_columns(pl.col("cdr3aa").str.len_chars().alias("cdr3Length"))
        .with_columns(
            (pl.col("v_gene") + "_" + pl.col("cdr3aa") + "_" + pl.col("j_gene"))
            .alias("clone")
        )
    )
    
    # Filter by CDR3 length
    query = query.filter(
        (pl.col('cdr3Length') >= min_cdr3_length) & 
        (pl.col('cdr3Length') < max_cdr3_length)
    )
    
    # Filter by CDR3 structure - MUST start with C and end with F
    query = query.filter(
        pl.col('cdr3aa').str.starts_with('C') & 
        pl.col('cdr3aa').str.ends_with('F')
    )
    
    # Select final columns and collect
    df = query.select([
        'cdr3aa', 'count', 'v_gene', 'j_gene', 
        'sample_id', 'cdr3Length', 'clone'
    ]).collect()
    
    return df

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
        TCRKIT_PREFIX = config.get('tcrkit', {}).get('output_prefix', 'gliph_fold')
        
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
        TCRKIT_PREFIX = 'gliph_fold'
    
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
    
    for i in range(1, N_FOLDS+1):
        fold_output_dir = f"./gliph_fold_{i+1}"
        print(f"\nCreating GLIPH2 files for fold {i+1} in {fold_output_dir}...")
        
        # This creates the gliph_fold_X directory AND the GLIPH2 input files!
        gliph_seropositive = seropositive[i].to_gliph(
            output=fold_output_dir,
            output_prefix = f'{TCRKIT_PREFIX}_1',
            condition=TCRKIT_CONDITION,
            reference_mode="other_sample",
            reference_data=seronegative[i],
            data_name=TCRKIT_CONDITION,
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