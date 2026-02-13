import pandas as pd
import yaml
import os
from typing import Optional


def clean_gliph_data(gliph_name: str, file_dir: Optional[str] = None):
    """
    Clean a GLIPH2 cluster file by removing additional spaces.

    Args:
        gliph_name: base name of the file (without .csv)
        file_dir: directory containing the file
    """

    if gliph_name is None:
        raise ValueError("GLIPH2 cluster file name must be provided")

    if file_dir is None:
        gliph_file_base = f"{gliph_name}.csv"
        gliph_file_clean = f"{gliph_name}_clean.csv"
    else:
        gliph_file_base = os.path.join(file_dir, f"{gliph_name}.csv")
        gliph_file_clean = f"{gliph_name}_clean.csv"

    # Load original file
    gliph_base = pd.read_csv(
        gliph_file_base,
        sep=",",
        skipinitialspace=True,
        on_bad_lines="warn",
        engine="python"
    )

    print(f"Initial shape of the dataframe ({gliph_name}): {gliph_base.shape}")

    # Save cleaned file
    gliph_base.to_csv(gliph_file_clean, index=False)

    # Reload cleaned file
    gliph_clean = pd.read_csv(
        gliph_file_clean,
        sep=",",
        skipinitialspace=True,
        on_bad_lines="skip",
        engine="python"
    )

    print(f"Shape of the clean dataframe ({gliph_name}): {gliph_clean.shape}")
    print("-" * 50)


if __name__ == "__main__":

    # Load config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    n_folds = config["n_folds"]
    output_prefix = config["tcrkit"]["output_prefix"]

    for fold in range(1, n_folds + 1):
        fold_dir = f"{output_prefix}_{fold}"
        gliph_name = f"{output_prefix}_{fold}_cluster"

        if not os.path.exists(os.path.join(fold_dir, f"{gliph_name}.csv")):
            print(f"WARNING: File not found for fold {fold}")
            continue

        clean_gliph_data(gliph_name, fold_dir)