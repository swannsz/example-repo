import os

dir_path = os.getcwd()
number_gliph_folder = len(glob.glob(f'{dir_path}/gliph_fold_*'))

for fold in range(number_gliph_folder) :
    # Read the file
    with open('gliph_fold_diabetes.cfg', 'r') as file:
        content = file.read()

    # Remove all occurrences of 'test_gfin/' using regex
    content = content.replace(f'gliph_fold_{fold+1}/', '')

    # Write back to the file
    with open('your_file.cfg', 'w') as file:
        file.write(content)