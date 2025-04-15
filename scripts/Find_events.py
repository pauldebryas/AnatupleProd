import os
import uproot
import json
import numpy as np

def list_root_files(directory):
    """Lists all .root files in the specified directory."""
    return [f for f in os.listdir(directory) if f.endswith(".root")]

def extract_events(root_file):
    """Extracts the 'event' array from the 'Events;1' tree in a ROOT file."""
    with uproot.open(root_file) as file:
        try:
            tree = file["Events;1"]  # Access the tree
            events = tree["event"].array(library="np")  # Extract event array as numpy
            return events
        except Exception as e:
            print(f"Error reading {root_file}: {e}")
            return np.array([])

def load_reference_events(json_file):
    """Loads the reference event array from a JSON file."""
    with open(json_file, "r") as f:
        data = json.load(f)
        return np.array(data)  # Ensure it's a NumPy array

def main(directory, json_file):
    """Main function to compare events in ROOT files with reference events."""
    root_files = list_root_files(directory)
    reference_events = load_reference_events(json_file)
    
    for root_file in root_files:
        root_path = os.path.join(directory, root_file)
        array_root = extract_events(root_path)
        
        common_events = np.intersect1d(array_root, reference_events)  # Find common events
        
        print(f"File: {root_file}")
        print(f"Common Events: {common_events.tolist()}")
        print("-" * 40)

if __name__ == "__main__":
    directory = "/eos/cms/store/group/phys_higgs/HLepRare/HTT_skim_v1/Run2_2018/DYJetsToLL_M-50-amcatnloFXFX"  # Change to your actual directory
    json_file = "/afs/cern.ch/user/p/pdebryas/HNL_analysis/Analysis/AnatupleProd_LLFF/scripts/saved_event.json"   # Change to your actual JSON file
    main(directory, json_file)
