# GLIPH2 TCR Analysis Snakemake Workflow

## Overview

This Snakemake workflow automates the complete GLIPH2 TCR clustering and motif classification pipeline.

## Workflow Steps

1. **Create folders** (`create_files.py`): Generates analysis folders for each fold
2. **Run GLIPH2**: Executes GLIPH2 clustering in each folder
3. **Clean data** (`clean_gliph_data.py`): Cleans cluster CSV files
4. **Classification** (`tcr_motif_classification.py`): Performs final classification

## Prerequisites

```bash
# Install Snakemake
conda install -c conda-forge snakemake

# Or using pip
pip install snakemake
```

## Required Files

Ensure these files are in your working directory:
- `create_files.py`
- `clean_gliph_data.py`
- `tcr_motif_classification.py`
- `clean_metadata.csv`
- `Snakefile`
- `config.yaml` (optional)

## Configuration

Edit the `Snakefile` or `config.yaml` to set:
- `N_FOLDS`: Number of cross-validation folds (default: 5)
- `GLIPH2_PATH`: Path to your GLIPH2 installation
- `METADATA_FILE`: Path to your metadata file

## Usage

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

## Workflow Diagram

```
clean_metadata.csv
        ↓
create_files.py → folders_created.done
        ↓
[gliph_fold_1, gliph_fold_2, ..., gliph_fold_5]
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

## Advanced Options

### Run with specific number of cores
```bash
snakemake --cores 8
```

### Use cluster execution (SLURM example)
```bash
snakemake --cluster "sbatch -c {threads} -t 24:00:00" --jobs 10
```

### Generate execution report
```bash
snakemake --report report.html
```

## Troubleshooting

### Check logs
All logs are stored in `logs/` directory:
- `logs/create_folders.log`
- `logs/gliph2_fold_[1-5].log`
- `logs/clean_gliph_data.log`
- `logs/tcr_classification.log`

### Force re-run specific rule
```bash
snakemake --forcerun rule_name --cores 4
```

### Unlock workflow (if interrupted)
```bash
snakemake --unlock
```

## Notes

- **GLIPH2 Command**: You need to modify the `run_gliph2_clustering` rule to match your GLIPH2 installation and input format
- **Screen sessions**: The original workflow mentioned using `screen` - Snakemake handles job management, so this is no longer necessary
- **Parallel execution**: GLIPH2 clustering for each fold runs in parallel (up to `--cores` limit)

## Customization

You can extend this workflow by:
1. Adding quality control steps
2. Adding visualization rules
3. Adding parameter optimization
4. Integrating with cluster schedulers (SLURM, SGE, etc.)