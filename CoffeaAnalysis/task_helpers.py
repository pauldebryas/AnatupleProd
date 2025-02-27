import os
import shutil 
import pickle
from itertools import islice

def split_list(files, nSublist):
    """Splits a list of files into 3 sublists with approximately equal size."""
    avg = len(files) // nSublist
    remainder = len(files) % nSublist
    result = []
    iterator = iter(files)
    
    for i in range(nSublist):
        size = avg + (1 if i < remainder else 0)
        result.append(list(islice(iterator, size)))

    return result

def change_paths(source_files):
    new_source_files = []
    for file in source_files:
        new_path = file.replace("/eos/cms/store/group/phys_higgs/HLepRare", "/eos/user/p/pdebryas/HNL")
        new_source_files.append(new_path)
    return new_source_files

def get_size_in_gb(path):
    total_size = 0
    
    # Iterate through all files in the directory
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Check if the file exists before getting its size
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    
    # Convert bytes to gigabytes
    return round(total_size / (1024 ** 3))

def files_from_dir(d):
    '''Returns a list of all ROOT files in the passed directory.
    '''
    files = os.listdir(d)
    return [os.path.join(d,f) for f in files if f.endswith('.root')]

def HNL_from_dir(d, mass):
    '''Returns a list of all ROOT files in the passed directory.
    '''
    files = os.listdir(d)
    return [os.path.join(d,f) for f in files if f.endswith(str(mass)+'.root')]

def one_file_from_dir(d):
    '''Returns a list of all ROOT files in the passed directory.
    '''
    files = os.listdir(d)
    files = files[:1]
    return [os.path.join(d,f) for f in files if f.endswith('.root')]

def files_from_dirs(dirs):
    '''Returns a list of all ROOT files from the directories in the passed list.
    '''
    files = []
    for d in dirs:
        files += files_from_dir(d)
    return files

def files_from_path(path):
    '''Returns a list of all ROOT files from a path.
    '''
    if path.endswith('.root'):
        files = []
        files.append(path)
        return files
    else:
        files = os.listdir(path)
        return [os.path.join(path,f) for f in files if (f.endswith('.root') and not f.startswith('.'))]

def cleanup_ds(ds, output_tmp_folder, output_root_folder):
    ''' Hadd and then move anatuple rootfiles from tmp folder to CoffeaAnalysis/results
    '''
    debug_mode = True

    if debug_mode == True:
        print('Apply cleanup_ds to '+ ds)

    if not os.path.exists(output_tmp_folder):
        os.makedirs(output_tmp_folder)

    files = os.listdir(output_tmp_folder)
    filelist = []
    filelist_cmd = ''
    for file in files:
        if ds == file.split('_anatuple_')[0]:
            filelist.append(file)
            filelist_cmd = filelist_cmd + output_tmp_folder + file + ' '
            
    if len(filelist) == 0:
        if debug_mode == True:
            print("No file found for " + ds)
            # create a dummy output file 
            output_file_name = output_root_folder + '/' + ds + '_anatuple.root'
            import ROOT
            file = ROOT.TFile(output_file_name, "CREATE") 
            file.Close()
        return

    if len(filelist) == 1:
        filename = ds + '_anatuple.root'
        source = os.path.join(output_tmp_folder, filelist[0])
        destination = os.path.join(output_root_folder, filename)        
        if os.path.exists(destination):
            raise f'File {destination} exist already, cannot move {source}'
        if debug_mode == True:
            print('Running the folowing command:')
            print(f"shutil.move({source}, {destination})")
        shutil.move(source, destination)
        return

    if (len(filelist) > 1) & (len(filelist) < 1000):
        hadd_cmd = 'hadd -n 11 ' + output_root_folder + '/' + ds + '_anatuple.root ' + filelist_cmd 
        if debug_mode == True:
            print('Running the folowing command:')
            print(hadd_cmd)
        os.system(hadd_cmd)

        for file in filelist:
           if debug_mode == True:
               print('Run the folowing command:')
               print(f"os.remove({os.path.join(output_tmp_folder, file)})")
           os.remove(os.path.join(output_tmp_folder, file))
        return

    if (len(filelist) >= 1000):
        split_int = int(len(filelist)/2)

        filelist_cmd_1 = ''
        filelist_cmd_2 = ''

        for file in files[0:split_int]:
            if ds == file.split('_anatuple_')[0]:
                filelist_cmd_1 = filelist_cmd_1 + output_tmp_folder + file + ' '

        for file in files[split_int:]:
            if ds == file.split('_anatuple_')[0]:
                filelist_cmd_2 = filelist_cmd_2 + output_tmp_folder + file + ' '


        hadd_cmd_1 = 'hadd -n 11 ' + output_root_folder + '/' + ds + '_anatuple_0.root ' + filelist_cmd_1
        hadd_cmd_2 = 'hadd -n 11 ' + output_root_folder + '/' + ds + '_anatuple_1.root ' + filelist_cmd_2

        if debug_mode == True:
            print('Running the folowing command:')
            print(hadd_cmd_1)
            print(hadd_cmd_2)
        os.system(hadd_cmd_1)
        os.system(hadd_cmd_2)

        hadd_cmd_final = 'hadd -n 11 ' + output_root_folder + '/' + ds + '_anatuple.root '  + output_root_folder + '/' + ds + '_anatuple_0.root ' + output_root_folder + '/' + ds + '_anatuple_1.root '
        if debug_mode == True:
            print('Running the folowing command:')
            print(hadd_cmd_final)
        os.system(hadd_cmd_final)

        if debug_mode == True:
            print('Running the folowing command:')
            print(f"os.remove({os.path.join(output_tmp_folder, ds + '_anatuple_0.root')})")
        os.remove(os.path.join(output_root_folder, ds + '_anatuple_0.root'))
        if debug_mode == True:
            print('Running the folowing command:')
            print(f"os.remove({os.path.join(output_tmp_folder, ds + '_anatuple_1.root')})")
        os.remove(os.path.join(output_root_folder, ds + '_anatuple_1.root'))

        for file in filelist:
           if debug_mode == True:
               print('Run the folowing command:')
               print(f"os.remove({os.path.join(output_tmp_folder, file)})")
           os.remove(os.path.join(output_tmp_folder, file))
           
        return
    
    return

def merge_pkl_files(input_files, output_file):

    print(f"Merge {len(input_files)} files into {output_file}")

    if not input_files:
        print("No input files provided.")
        return

    merged_data = None  # Initialize as None to take the structure from the first file

    for file in input_files:
        if not os.path.exists(file):
            print(f"Warning: {file} not found, skipping...")
            continue

        with open(file, "rb") as f:
            try:
                data = pickle.load(f)
                if merged_data is None:
                    # Initialize merged_data with the structure of the first file
                    merged_data = {key: {subkey: value for subkey, value in subdict.items()} 
                                   for key, subdict in data.items()}
                else:
                    # Sum values from subsequent files
                    for key in data:
                        for subkey in data[key]:
                            merged_data[key][subkey] += data[key][subkey]

            except Exception as e:
                print(f"Error loading {file}: {e}")

    # Save the merged result
    if merged_data is not None:
        with open(output_file, "wb") as f:
            pickle.dump(merged_data, f)
