import os
import argparse
import yaml


def clean_config_file(fold_number: int, config: dict, condition: str = None):
    """
    Clean GLIPH2 config file by removing directory prefixes.

    Args:
        fold_number: The fold number to process
        config: Loaded YAML config
        condition: Optional override for TCRkit condition
    """

    output_prefix = config["tcrkit"]["output_prefix"]
    default_condition = config["tcrkit"].get("condition", "diabetes")

    if condition is None:
        condition = default_condition

    fold_dir = f"{output_prefix}_{fold_number}"
    input_file = os.path.join(fold_dir, f"gliph_config_{condition}.cfg")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Config file not found: {input_file}")

    # Read file
    with open(input_file, "r") as file:
        content = file.read()

    # Remove directory prefixes
    content = content.replace(f"{fold_dir}/", "")

    # Write cleaned content
    with open(input_file, "w") as file:
        file.write(content)

    print(f"✓ Cleaned config file: {input_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Clean GLIPH2 config files")
    parser.add_argument("--fold", type=int, required=True, help="Fold number")
    parser.add_argument("--condition", type=str, help="Optional TCRkit condition override")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")

    args = parser.parse_args()

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    clean_config_file(args.fold, config, args.condition)