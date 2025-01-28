import os
import shutil 

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
    debug_mode = False

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