# GLIPH2 TCR Analysis Snakemake Workflow

## Overview
![pipeline complète](./not%20used%20now/ml_strategie.png)

This Snakemake workflow automates the complete GLIPH2 TCR clustering and motif classification pipeline.
## Workflow Steps

1. **Create folders** (`create_files_updated.py`): Generates analysis folders for each fold
2. **Clean GLIPH2 config file** (`clean_gliph_config.py`): Remove in each folders created by (1.) the folder name for each {file}.txt. _It exist because of the rep.to_gliph() function of tcrkit, the final goal is to remove it_
3. **Run GLIPH2**: Executes GLIPH2 clustering in each folder
4. **Clean data** (`clean_gliph_data.py`): Cleans cluster CSV files _If csv files are too bigs there is problems when opening them, using this file permit to suppress the unnecessaries spaces and make the csv file readable_
5. **Classification** (`classif_updated.py`): Performs final classification

## Prerequisites

Install the tcrkit package and create a conda environnement (see on the tcrkit github how to do it)
```bash
# Install polars
pip install polars

# Install Snakemake
conda install -c conda-forge snakemake

# Or using pip
pip install snakemake
```

## Required Files

Ensure these files are in your working directory:
- `create_files_updated.py`
- `clean_gliph_data.py`
- `clean_gliph_config.py`
- `classif_updated.py`
- `clean_metadata.csv`
- `Snakefile`
- `config.yaml`

## Configuration

Edit the `config.yaml` to set:
- `N_FOLDS`: Number of cross-validation folds (default: 5)
- `GLIPH2_PATH`: Path to your GLIPH2 installation
- `METADATA_FILE`: Path to your metadata file
- and a lot of other parameters

## Usage

### If you use SLURM
adapt the classification.sbatch file to your needs
```bash
sbatch classification.sbatch
```

### Run the complete workflow
```bash
snakemake --cores 4
```

### Dry run (see what will be executed)
```bash
snakemake --dry-run --printshellcmds
```

### Run specific steps
```bash
# Only create folders
snakemake folders_created.done --cores 1

# Run up to cleaning
snakemake gliph_cleaning.done --cores 4
```

### Generate workflow diagram
```bash
snakemake --dag | dot -Tpng > workflow.png
```

### Clean all outputs
```bash
snakemake clean
```

### Generate execution report
```bash
snakemake --report report.html
```

## Workflow Diagram

```
clean_metadata.csv
        ↓
create_files.py → folders_created.done
        ↓
[gliph_fold_1, gliph_fold_2, ..., gliph_fold_5]
        ↓
clean_gliph_data.py → gliph_cleaning.done
        ↓
run_gliph2_clustering (parallel)
        ↓
[cluster files in each folder]
        ↓
clean_gliph_data.py → gliph_cleaning.done
        ↓
[*_cluster_clean.csv files]
        ↓
tcr_motif_classification.py → classification_results.done
```

## Troubleshooting

### Check logs
All logs are stored in `logs/` directory:
- `logs/create_folders.log`
- `logs/gliph2_fold_[1-5].log`
- `logs/clean_gliph_data.log`
- `logs/tcr_classification.log`

### Check out or err files (if using the classification.sbatch)

### Force re-run specific rule
```bash
snakemake --forcerun rule_name --cores 4
```

### Unlock workflow (if interrupted)
```bash
snakemake --unlock
```
