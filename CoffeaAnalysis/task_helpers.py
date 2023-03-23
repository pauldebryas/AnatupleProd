import os

def files_from_dir(d):
    '''Returns a list of all ROOT files in the passed directory.
    '''
    files = os.listdir(d)
    return ['/'.join([d, f]) for f in files if f.endswith('.root')]

def HNL_from_dir(d, mass):
    '''Returns a list of all ROOT files in the passed directory.
    '''
    files = os.listdir(d)
    return ['/'.join([d, f]) for f in files if f.endswith(str(mass)+'.root')]

def one_file_from_dir(d):
    '''Returns a list of all ROOT files in the passed directory.
    '''
    files = os.listdir(d)
    files = files[:1]
    return ['/'.join([d, f]) for f in files if f.endswith('.root')]

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
        return ['/'.join([path, f]) for f in files if (f.endswith('.root') and not f.startswith('.'))]

def cleanup_ds(ds, output_tmp_folder, output_root_folder):
    ''' Hadd and then move anatuple rootfiles from tmp folder to CoffeaAnalysis/results
    '''
    debug_mode = False

    if debug_mode == True:
        print('processing '+ ds)

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
        return

    if len(filelist) == 1:
        rename_cmd = 'mv ' + filelist_cmd + output_root_folder + '/' + ds + '_anatuple.root '
        if debug_mode == True:
            print('Running the folowing command:')
            print(rename_cmd)
        os.system(rename_cmd)
        return

    if len(filelist) > 1:

        hadd_cmd = 'hadd ' + output_root_folder + '/' + ds + '_anatuple.root ' + filelist_cmd
        if debug_mode == True:
            print('Running the folowing command:')
            print(hadd_cmd)
        os.system(hadd_cmd)

        rm_cmd = 'rm ' + filelist_cmd
        if debug_mode == True:
            print('Run the folowing command:')
            print(rm_cmd)
        os.system(rm_cmd)
    
    return