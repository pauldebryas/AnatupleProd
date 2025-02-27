import shutil
import os
from tqdm import tqdm

# List of source files
source_files = [
    "/eos/cms/store/group/phys_higgs/HLepRare/HTT_skim_v1/Run2_2018/TTToSemiLeptonic/nanoHTT_233.root",
    "/eos/cms/store/group/phys_higgs/HLepRare/HTT_skim_v1/Run2_2018/TTToSemiLeptonic/nanoHTT_237.root",
    "/eos/cms/store/group/phys_higgs/HLepRare/HTT_skim_v1/Run2_2018/TTToSemiLeptonic/nanoHTT_242.root",
    "/eos/cms/store/group/phys_higgs/HLepRare/HTT_skim_v1/Run2_2018/TTToSemiLeptonic/nanoHTT_262.root",
    "/eos/cms/store/group/phys_higgs/HLepRare/HTT_skim_v1/Run2_2018/TTToSemiLeptonic/nanoHTT_265.root",
    "/eos/cms/store/group/phys_higgs/HLepRare/HTT_skim_v1/Run2_2018/TTToSemiLeptonic/nanoHTT_272.root",
    "/eos/cms/store/group/phys_higgs/HLepRare/HTT_skim_v1/Run2_2018/TTToSemiLeptonic/nanoHTT_273.root"
]

# Destination directory
destination_dir = "/eos/user/p/pdebryas/HNL/HTT_skim_v1/Run2_2018/TTToSemiLeptonic/"

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Copy files with progress bar
for file in tqdm(source_files, desc="Copying files"):
    try:
        shutil.copy2(file, destination_dir)
        print(f"Copied: {file}")
    except Exception as e:
        print(f"Failed to copy {file}: {e}")

