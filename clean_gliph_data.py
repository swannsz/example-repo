import pandas as pd
from typing import Union, List, Dict, Optional, Any, Tuple
import os
import glob


def clean_gliph_data (gliph_name : str, file_dir : str = None) :
    """
    Clean a GLIPH2 cluster file by removing all the additionnal space.
    
    Args:
        gliph_name: name of the file to clean
        file_dir: name of the directory for the GLIPH2 cluster file to clean
    """
    if gliph_name is None:
        raise ValueError('GLIPH2 cluster file name path must be provided')
        
    if file_dir is None :
        gliph_file_base = f'{gliph_name}.csv'
    
    else :
        gliph_file_base = os.path.join(file_dir, f'{gliph_name}.csv')
    
    # Load data with pandas 
    gliph_base = pd.read_csv(
        gliph_file_base,
        sep=",",
        skipinitialspace=True,
        on_bad_lines="warn",
        engine="python")

    # Print the shape of the initial dataframe
    print(f'Initial shape of the dataframe : {gliph_base.shape}')
    
    # Export the new dataframe with a new name in the working directory
    gliph_file_clean = f'{gliph_name}_clean.csv'
    gliph_base.to_csv(gliph_file_clean, index=False)

    # Load data with pandas  
    gliph_clean = pd.read_csv(
        gliph_file_clean,
        sep=",",
        skipinitialspace=True,
        on_bad_lines="skip",
        engine="python")
    
    # Compare the shape of the clean dataframe with the old one
    print(f'Shape of the clean dataframe : {gliph_clean.shape}')

if __name__ == "__main__":
    # All files and directories starting with gliph_fold_ :
    dir_path = os.getcwd()
    number_gliph_folder = len(glob.glob(f'{dir_path}/gliph_fold_*'))
    
    for i in range(number_gliph_folder) :
        clean_gliph_data(f'gliph_fold_{i+1}_cluster', f'gliph_fold_{i+1}')