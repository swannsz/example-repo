"""
Snakemake workflow for GLIPH2 TCR clustering and motif classification
Configuration-driven version using config.yaml
"""

import os
import glob

# =============================================================================
# LOAD CONFIGURATION
# =============================================================================
configfile: "config.yaml"

# Extract frequently used config values
N_FOLDS = config["n_folds"]
FOLDS = [str(i) for i in range(1, N_FOLDS + 1)]
METADATA_FILE = config["paths"]["metadata"]
GLIPH2_PATH = config["paths"]["gliph2_bin"]
TCRKIT_CONDITION = config["tcrkit"]["condition"]

# =============================================================================
# RULES
# =============================================================================

rule all:
    """
    Final target: complete the entire workflow
    """
    input:
        "classification_results.done"

rule create_folders:
    """
    Step 1: Run create_files.py which:
    - Creates train/val/test CSV files
    - Creates gliph_fold_X directories via to_gliph()
    - Generates GLIPH2 input files and config files in each folder
    """
    input:
        script="create_files.py",
        metadata=METADATA_FILE,
        config_file="config.yaml"
    output:
        # The CSV files created
        train_files=expand("fold_train_{fold}.csv", fold=FOLDS),
        val_files=expand("fold_val_{fold}.csv", fold=FOLDS),
        test_file="test_data.csv",
        # Marker to indicate folders were created by to_gliph()
        folder_marker=touch("folders_created.done")
    params:
        config_path="config.yaml"
    log:
        "logs/create_folders.log"
    shell:
        """
        python {input.script} --config {params.config_path} > {log} 2>&1
        """

rule run_gliph2_clustering:
    """
    Step 2: Run GLIPH2 clustering in each fold directory
    
    CRITICAL: GLIPH2 creates temporary files with fixed names
    Therefore, each job MUST run in its own directory to avoid conflicts
    
    This rule runs once per fold - Snakemake handles parallelization
    Each job runs in its own gliph_fold_X directory
    """
    input:
        folder_marker="folders_created.done"
    output:
        cluster="gliph_fold_{fold}/gliph_fold_{fold}_cluster.csv"
    params:
        fold_dir="gliph_fold_{fold}",
        gliph_path=lambda wildcards: os.path.abspath(GLIPH2_PATH),  # Absolute path
        config_file=f"gliph_config_{TCRKIT_CONDITION}.cfg"  # Dynamic config filename
    log:
        "logs/gliph2_fold_{fold}.log"
    threads: config["resources"]["threads"]["gliph2"]
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 8000
    shell:
        """
        # Get absolute path to log file
        log_abs=$(readlink -f {log})
        
        echo "Running GLIPH2 for fold {wildcards.fold}..." > "$log_abs"
        echo "Fold directory: {params.fold_dir}" >> "$log_abs"
        echo "Config file: {params.config_file}" >> "$log_abs"
        echo "TCRkit condition: {TCRKIT_CONDITION}" >> "$log_abs"
        echo "Working directory will be: $(pwd)/{params.fold_dir}" >> "$log_abs"
        echo "" >> "$log_abs"
        
        # Change to fold directory
        cd {params.fold_dir}
        
        # Check if config file exists
        if [ ! -f "{params.config_file}" ]; then
            echo "ERROR: Config file not found: {params.config_file}" >> "$log_abs"
            echo "Current directory: $(pwd)" >> "$log_abs"
            echo "Files in directory:" >> "$log_abs"
            ls -la >> "$log_abs"
            exit 1
        fi
        
        # Show config file content for debugging
        echo "Config file content:" >> "$log_abs"
        cat {params.config_file} >> "$log_abs"
        echo "" >> "$log_abs"
        
        # Run GLIPH2 from inside the fold directory
        # -c flag specifies the config file
        echo "Executing: {params.gliph_path} -c {params.config_file}" >> "$log_abs"
        {params.gliph_path} -c {params.config_file} >> "$log_abs" 2>&1
        
        exit_code=$?
        
        # Check if GLIPH2 succeeded
        if [ $exit_code -ne 0 ]; then
            echo "ERROR: GLIPH2 failed with exit code $exit_code" >> "$log_abs"
            exit $exit_code
        fi
        
        # Check if output was created
        if [ ! -f "gliph_fold_{wildcards.fold}_cluster.csv" ]; then
            echo "ERROR: GLIPH2 did not create expected output file" >> "$log_abs"
            echo "Files created:" >> "$log_abs"
            ls -la >> "$log_abs"
            exit 1
        fi
        
        echo "" >> "$log_abs"
        echo "✓ GLIPH2 completed successfully for fold {wildcards.fold}" >> "$log_abs"
        echo "Output file: gliph_fold_{wildcards.fold}_cluster.csv" >> "$log_abs"
        """

rule aggregate_gliph_clusters:
    """
    Step 2b: Ensure all GLIPH2 clustering is complete
    This waits for all parallel GLIPH2 jobs to finish
    """
    input:
        expand("gliph_fold_{fold}/gliph_fold_{fold}_cluster.csv", fold=FOLDS)
    output:
        touch("all_gliph_clustering.done")
    log:
        "logs/aggregate_gliph.log"
    shell:
        """
        echo "All GLIPH2 clustering completed for {N_FOLDS} folds" > {log}
        echo "" >> {log}
        
        for fold in {FOLDS}; do
            cluster_file="gliph_fold_${{fold}}/gliph_fold_${{fold}}_cluster.csv"
            if [ -f "$cluster_file" ]; then
                lines=$(wc -l < "$cluster_file")
                size=$(du -h "$cluster_file" | cut -f1)
                echo "✓ Fold $fold: $cluster_file ($lines lines, $size)" >> {log}
            else
                echo "✗ Fold $fold: MISSING!" >> {log}
            fi
        done
        
        echo "" >> {log}
        echo "Summary:" >> {log}
        echo "  Total folds: {N_FOLDS}" >> {log}
        echo "  Completed: $(ls gliph_fold_*/gliph_fold_*_cluster.csv 2>/dev/null | wc -l)" >> {log}
        """

rule clean_gliph_data:
    """
    Step 3: Clean GLIPH2 cluster files
    No config needed - automatically detects all gliph_fold_* folders
    """
    input:
        script="clean_gliph_data.py",
        all_clusters="all_gliph_clustering.done"
    output:
        cleaned_files=expand("gliph_fold_{fold}_cluster_clean.csv", fold=FOLDS),
        done_marker=touch("gliph_cleaning.done")
    log:
        "logs/clean_gliph_data.log"
    shell:
        """
        python {input.script} > {log} 2>&1
        """

rule tcr_classification:
    """
    Step 4: Run TCR motif classification
    """
    input:
        script="tcr_motif_classification.py",
        cleaned_flag="gliph_cleaning.done",
        clusters=expand("gliph_fold_{fold}_cluster_clean.csv", fold=FOLDS),
        train=expand("fold_train_{fold}.csv", fold=FOLDS),
        val=expand("fold_val_{fold}.csv", fold=FOLDS),
        test="test_data.csv",
        config_file="config.yaml"
    output:
        touch("classification_results.done")
    params:
        config_path="config.yaml"
    log:
        "logs/tcr_classification.log"
    threads: config["resources"]["threads"]["classification"]
    resources:
        mem_mb=16000
    shell:
        """
        python {input.script} --config {params.config_path} > {log} 2>&1
        """

# =============================================================================
# UTILITY RULES
# =============================================================================
rule list_outputs:
    """
    List all output files and their status
    """
    params:
        config_filename=f"gliph_config_{TCRKIT_CONDITION}.cfg"
    run:
        import os
        import glob
        
        print("\n" + "="*60)
        print("Output Files Status")
        print("="*60)
        print(f"TCRkit condition: {TCRKIT_CONDITION}")
        print(f"GLIPH2 config filename: {params.config_filename}")
        print("="*60 + "\n")
        
        # Check CSV files
        print("Training files:")
        for fold in FOLDS:
            fname = f"fold_train_{fold}.csv"
            status = "✓" if os.path.exists(fname) else "✗"
            print(f"  {status} {fname}")
        
        print("\nValidation files:")
        for fold in FOLDS:
            fname = f"fold_val_{fold}.csv"
            status = "✓" if os.path.exists(fname) else "✗"
            print(f"  {status} {fname}")
        
        print("\nTest file:")
        fname = "test_data.csv"
        status = "✓" if os.path.exists(fname) else "✗"
        print(f"  {status} {fname}")
        
        # Check GLIPH folders
        print("\nGLIPH2 folders:")
        for fold in FOLDS:
            dirname = f"gliph_fold_{fold}"
            status = "✓" if os.path.isdir(dirname) else "✗"
            print(f"  {status} {dirname}/")
            
            if os.path.isdir(dirname):
                # Check for config file
                config_file = f"{dirname}/{params.config_filename}"
                config_status = "✓" if os.path.exists(config_file) else "✗"
                print(f"      {config_status} {params.config_filename}")
                
                # Count files
                num_files = len([f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))])
                print(f"      ({num_files} files total)")
        
        # Check cluster files
        print("\nGLIPH2 cluster files:")
        for fold in FOLDS:
            fname = f"gliph_fold_{fold}/gliph_fold_{fold}_cluster.csv"
            status = "✓" if os.path.exists(fname) else "✗"
            if os.path.exists(fname):
                size = os.path.getsize(fname)
                lines = sum(1 for line in open(fname))
                print(f"  {status} {fname} ({lines} lines, {size} bytes)")
            else:
                print(f"  {status} {fname}")
        
        # Check cleaned files
        print("\nCleaned cluster files:")
        for fold in FOLDS:
            fname = f"gliph_fold_{fold}_cluster_clean.csv"
            status = "✓" if os.path.exists(fname) else "✗"
            print(f"  {status} {fname}")
        
        # Check final results
        print("\nFinal results:")
        status = "✓" if os.path.exists("classification_results.done") else "✗"
        print(f"  {status} classification_results.done")
        
        print("\n" + "="*60 + "\n")

rule validate_config:
    """
    Validate the configuration file
    """
    params:
        config_filename=f"gliph_config_{TCRKIT_CONDITION}.cfg"
    run:
        print("\n" + "="*60)
        print("Configuration Validation")
        print("="*60 + "\n")
        
        print(f"Number of folds: {config['n_folds']}")
        print(f"Metadata file: {config['paths']['metadata']}")
        print(f"GLIPH2 binary: {config['paths']['gliph2_bin']}")
        print(f"Cell subset type: {config['metadata']['cell_subset_type']}")
        print(f"TCRkit condition: {TCRKIT_CONDITION}")
        print(f"GLIPH2 config filename: {params.config_filename}")
        print(f"GLIPH2 p-value threshold: {config['gliph2']['threshold_pvalue']}")
        
        import os
        
        # Check metadata file
        if not os.path.exists(config['paths']['metadata']):
            print(f"\n⚠️  WARNING: Metadata file not found: {config['paths']['metadata']}")
        else:
            print(f"\n✓ Metadata file exists")
        
        # Check GLIPH2 binary
        gliph2_path = config['paths']['gliph2_bin']
        if not os.path.exists(gliph2_path):
            print(f"⚠️  WARNING: GLIPH2 binary not found: {gliph2_path}")
            print(f"   Please update 'paths.gliph2_bin' in config.yaml")
        elif not os.access(gliph2_path, os.X_OK):
            print(f"⚠️  WARNING: GLIPH2 binary not executable: {gliph2_path}")
            print(f"   Run: chmod +x {gliph2_path}")
        else:
            print(f"✓ GLIPH2 binary exists and is executable")
        
        print("\n" + "="*60 + "\n")

rule report:
    """
    Generate a summary report of the workflow
    """
    input:
        "classification_results.done"
    output:
        "workflow_report.txt"
    run:
        import os
        import glob
        from datetime import datetime
        
        with open(output[0], 'w') as f:
            f.write("="*60 + "\n")
            f.write("GLIPH2 TCR Analysis Workflow Report\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Configuration:\n")
            f.write(f"  - Number of folds: {N_FOLDS}\n")
            f.write(f"  - Metadata file: {METADATA_FILE}\n")
            f.write(f"  - Cell subset: {config['metadata']['cell_subset_type']}\n")
            f.write(f"  - GLIPH2 binary: {GLIPH2_PATH}\n")
            f.write(f"  - TCRkit condition: {TCRKIT_CONDITION}\n\n")
            
            f.write("Output files created:\n")
            f.write(f"  - Training files: {len(glob.glob('fold_train_*.csv'))}\n")
            f.write(f"  - Validation files: {len(glob.glob('fold_val_*.csv'))}\n")
            f.write(f"  - GLIPH2 folders: {len(glob.glob('gliph_fold_*'))}\n")
            f.write(f"  - Cleaned cluster files: {len(glob.glob('*_cluster_clean.csv'))}\n\n")
            
            f.write("Workflow completed successfully!\n")
            f.write("="*60 + "\n")
        
        print(f"✓ Report saved to {output[0]}")