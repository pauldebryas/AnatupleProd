
import os
import uproot
import numpy as np

# ------------------------- Parameters -------------------------
period= '2016'
tag = 'AddJETcorr'
test = False
# --------------------------------------------------------------

def get_common_files(folder1, folder2):
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    return sorted(list(files1.intersection(files2)))

def rename_branches(branches, prefix_old, prefix_new):
    return {name: name.replace(prefix_old, prefix_new, 1) for name in branches if name.startswith(prefix_old)}

def filter_branches(branches):
    return {k: v for k, v in branches.items() if not (k.startswith("weightcorr_") or k.startswith("nAdditional"))}

def merge_root_files(file1, file2, output_file):
    with uproot.open(file1) as f1, uproot.open(file2) as f2:
        if "Events;1" not in f1 or "Events;1" not in f2:
            print(f"Error: 'Events;1' tree not found in {file1} or {file2}")
            return
        
        tree1 = f1["Events;1"]
        tree2 = f2["Events;1"]
        
        # Get branch names and rename
        rename_map_t1 = rename_branches(tree1.keys(), "Electron", "Lepton")
        rename_map_t2 = rename_branches(tree2.keys(), "Muon", "Lepton")
        
        # Read trees into dictionaries
        data_t1 = tree1.arrays(library="np")
        data_t2 = tree2.arrays(library="np")
        
        # Rename branches
        for old_name, new_name in rename_map_t1.items():
            data_t1[new_name] = data_t1.pop(old_name)
        for old_name, new_name in rename_map_t2.items():
            data_t2[new_name] = data_t2.pop(old_name)
        
        # Filter unwanted branches
        data_t1 = {k: v for k, v in data_t1.items() if not (k.startswith("weightcorr_") or k.startswith("nAdditional"))}
        data_t2 = {k: v for k, v in data_t2.items() if not (k.startswith("weightcorr_") or k.startswith("nAdditional"))}
        
        # Merge the dictionaries
        merged_data = {k: np.concatenate((data_t1[k], data_t2[k])) for k in set(data_t1) & set(data_t2)}
        
        # Write to new ROOT file
        with uproot.recreate(output_file) as out_file:
            out_file["Events"] = merged_data

def main():
    eos_path = f'/eos/user/p/pdebryas/HNL/anatuple/{period}/{tag}/'
    folder_tee = os.path.join(eos_path, 'tee', 'anatuple')
    folder_tmm = os.path.join(eos_path, 'tmm', 'anatuple')
    output_folder = os.path.join(eos_path, 'tll', 'anatuple')
    os.makedirs(output_folder, exist_ok=True)
    
    common_files = get_common_files(folder_tee, folder_tmm)
    if test == True:
        common_files = ['GluGluHToTauTau_anatuple.root']
    
    for file_name in common_files:
        file_tee = os.path.join(folder_tee, file_name)
        file_tmm = os.path.join(folder_tmm, file_name)
        output_file = os.path.join(output_folder, file_name)
        print(f"Merging {file_name}...")
        merge_root_files(file_tee, file_tmm, output_file)

    HLT_Electron = {
        '2018': 'EGamma',
        '2017': 'SingleElectron',
        '2016': 'SingleElectron',
        '2016_HIPM': 'SingleElectron'
    }
    if test == False:
        print(f"Merging {HLT_Electron[period]}_{period} and SingleMuon_{period}...")
        data_file_tee = os.path.join(folder_tee, f'{HLT_Electron[period]}_{period}_anatuple.root')
        data_file_tmm = os.path.join(folder_tmm, f'SingleMuon_{period}_anatuple.root')
        output_file = os.path.join(output_folder,f'SingleMuonAnd{HLT_Electron[period]}_{period}_anatuple.root')
        merge_root_files(data_file_tee, data_file_tmm, output_file)
    
if __name__ == "__main__":
    main()

