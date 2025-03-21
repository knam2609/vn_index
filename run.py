import subprocess

# Define folder and notebook file names
folder = 'notebooks/'
file_names = ['scape', 'vn_index_preprocessing', 'external_data', 'external_preprecossing', 'merge', 'lstm_model']
file_type = '.ipynb'

for fn in file_names:
    file_path = folder + fn + file_type
    print(f"Running {file_path}...")
    subprocess.run(
        ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", file_path])

print("All scripts executed.")
