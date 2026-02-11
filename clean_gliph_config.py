import os
import glob
import argparse

def clean_config_file(fold_number: int, condition: str = "diabetes"):
    """
    Clean GLIPH2 config file by removing directory prefixes.
    
    Args:
        fold_number: The fold number to process
        condition: The TCRkit condition name (e.g., 'diabetes')
    """
    fold_dir = f'gliph_fold_{fold_number}'
    input_file = os.path.join(fold_dir, f'gliph_config_{condition}.cfg')
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Config file not found: {input_file}")
    
    # Read the file
    with open(input_file, 'r') as file:
        content = file.read()
    
    # Remove directory prefixes (e.g., 'gliph_fold_1/' -> '')
    content = content.replace(f'{fold_dir}/', '')
    
    # Write back to the same file
    with open(input_file, 'w') as file:
        file.write(content)
    
    print(f"✓ Cleaned config file: {input_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean GLIPH2 config files')
    parser.add_argument('--fold', type=int, required=True, help='Fold number to clean')
    parser.add_argument('--condition', type=str, default='diabetes', help='TCRkit condition')
    
    args = parser.parse_args()
    clean_config_file(args.fold, args.condition)