# run_all.py
import subprocess

scripts = [
    "vn_index_scripts/scrape_vn_index.py",
    "vn_index_scripts/vn_index_preprocessing.py",    
    "external_data_scripts/get_external_data.py",
    "external_data_scripts/external_preprocessing.py",
    "vn_30_scripts/vn30_preprocessing.py",
    "vn_30_scripts/vn30f1_preprocessing.py",
    "vn_30_list_scripts/scrape_vn30_list.py",
    "vn_30_list_scripts/vn30_list_preprocessing.py",
    "vn_index_scripts/merge_vn_index.py",
    "vn_30_scripts/merge_vn30.py",
    "vn_30_scripts/merge_vn30f1.py",
    "change_scripts/change_preprocessing.py",
    "change_scripts/merge_change.py"
]

for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python", f'scripts/{script}'], check=True)
