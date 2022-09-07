import json
import law
import luigi
import os
import shutil
import yaml
import re
import numpy as np

from run_tools.grid_helper_tasks import CreateVomsProxy
from run_tools.law_customizations import Task, HTCondorWorkflow
from run_tools.sh_tools import sh_call, xrd_copy

# Base task inherited by all other classes (all parameters in there)
class BaseTask(Task):
    dataset_tier = luigi.Parameter(default='nanoAOD')
    ignore_missing_samples = luigi.BoolParameter(default=False)
    max_size_limit = luigi.Parameter(default='20000000000') # [bytes] Limit for the sum of all the file size in a single batch

# Task to create a json file in which are stored all the dataset info (size, nevents, DAS path for each files) specified in config/samples_*.yalm input file
# LocalWorkflow ==> this task that should run locally 
class CreateDatasetInfos(BaseTask, law.LocalWorkflow):

    # requires VomsProxy to read files in DAS (same during workflow)
    def workflow_requires(self):
        return { "proxy": CreateVomsProxy.req(self) }
    
    def requires(self):
        return self.workflow_requires()

    # Branch map of the task: dataset in the input config file <->  1 json file with all the dataset info
    def create_branch_map(self):
        # read config/samples_*.yalm input file
        self.load_sample_configs()
        branch_index = 0
        branches = {}
        missing_samples = []
        for period, samples in self.samples.items():
            for sample_name in sorted(samples.keys()):
                das_dataset = samples[sample_name].get(self.dataset_tier, None)
                sampleType = samples[sample_name].get('sampleType', None)
                if sampleType is None or len(sampleType) == 0:
                    self.publish_message("Missing sampleType for sample: {}".format(sample_name))
                    raise RuntimeError("Missing sampleType has been detected.")
                if das_dataset is None or len(das_dataset) == 0:
                    missing_samples.append(f'{period}:{sample_name}')
                else:
                    branches[branch_index] = (sample_name, period, das_dataset, sampleType)
                branch_index += 1
        if len(missing_samples) > 0:
            self.publish_message("Missing samples: {}".format(', '.join(missing_samples)))
            if not self.ignore_missing_samples:
                raise RuntimeError("Missing samples has been detected.")
        if branches == {}:
            self.publish_message("CreateDatasetInfos Warning: branches are empty")
        return branches

    # output path for the json file
    def output(self):
        sample_name, period, das_dataset, sampleType = self.branch_data
        sample_out = os.path.join(self.central_path(), self.version, f'inputs_{self.dataset_tier}', period, sampleType, sample_name + '.json' )
        return law.LocalFileTarget(sample_out)

    # Find and store dataset info using dasgoclient command
    def run(self):
        sample_name, period, das_dataset, sampleType = self.branch_data
        self.publish_message(f'sameple_name={sample_name}, period={period}, das_dataset={das_dataset}')
        result = {}
        result['das_dataset'] = das_dataset
        result['size'] = 0
        result['nevents'] = 0
        result['files'] = []
        # use the following command to look at the files corresonding to the data sample: dasgoclient -json -query 'file dataset= ...' 
        _, output = sh_call(['dasgoclient', '-json', '-query', f'file dataset={das_dataset}'], catch_stdout=True)
        file_entries = json.loads(output)
        for file_entry in file_entries:
            if len(file_entry['file']) != 1:
                raise RuntimeError("Invalid file entry")
            out_entry = {
                'name': file_entry['file'][0]['name'],
                'size': file_entry['file'][0]['size'],
                'nevents': file_entry['file'][0]['nevents'],
                'adler32': file_entry['file'][0]['adler32'],
            }
            result['size'] += out_entry['size']
            result['nevents'] += out_entry['nevents']
            result['files'].append(out_entry)
        self.output().dump(result, indent=2)

# Task to create skimmed root files for each dataset specified in config/samples_*.yalm input file (by batch, limited by max_size_limit param)
# LocalWorkflow ==> this task that should be run with HTCondor workflow, or locally using "--CreateNanoSkims-workflow local" argument
class CreateNanoSkims(BaseTask, HTCondorWorkflow, law.LocalWorkflow):

    def __init__(self, *args, **kwargs):
        super(CreateNanoSkims, self).__init__(*args, **kwargs)
        self._cache_branches = False

    # requires json file with dataset info (CreateDatasetInfos task) and VOMS proxy to copy files locally.
    def workflow_requires(self):
        return {"proxy" : CreateVomsProxy.req(self), "dataset_info": CreateDatasetInfos.req(self, workflow='local') }

    # Branch sample_id is needed as there is no 1 to 1 correlation between CreateDatasetInfos task branches and CreateNanoSkims task branches
    def requires(self):
        sample_name_raw, sample_id, period, das_dataset, sampleType, input_file_remotes, alder_list = self.branch_data
        return {"proxy" : CreateVomsProxy.req(self), "dataset_info": CreateDatasetInfos.req(self, workflow='local', branch=sample_id) }

    # create 1 to 1 correlation if max_size_limit < size of all the files, otherwise create batches of files to be processed 
    def create_branch_map(self):
        self.load_sample_configs()
        branch_id = 0
        # id of the corresponding branch in CreateDatasetInfos
        sample_id = 0
        branches = {}
        missing_samples = []

        if not CreateDatasetInfos.req(self, workflow='local', branch= sample_id).output().exists():
            return { 0: None }
        
        for period, samples in self.samples.items():
            for sample_name in sorted(samples.keys()):

                sampleType = samples[sample_name].get('sampleType', None)
                if sampleType is None or len(sampleType) == 0:
                    self.publish_message("Missing sampleType for sample: {}".format(sample_name))
                    raise RuntimeError("Missing sampleType has been detected.")
                
                das_dataset = samples[sample_name].get(self.dataset_tier, None)
                if das_dataset is None or len(das_dataset) == 0:
                    missing_samples.append(f'{period}:{sample_name}')
                else:
                    # read the json file
                    sample_config_path = os.path.join(self.central_path(), self.version, f'inputs_{self.dataset_tier}', period, sampleType, sample_name + '.json' )
                    with open(sample_config_path, 'r') as f:
                        sample_config = json.load(f)
                    
                    # variables needed for each batch
                    input_file_remote=[]
                    alder_remote = []
                    for input_file in sample_config['files']:
                        input_file_remote.append(input_file['name'])
                        alder_remote.append(input_file.get('adler32', None))

                    if sample_config['size'] is None:
                        self.publish_message("Missing datafile size information in samples: {}: careful, it might crash if size to big !!".format(sample_name))
                        branches[branch_id] = (sample_name, sample_id, period, das_dataset, sampleType, input_file_remote, alder_remote)
                        branch_id += 1
                    else:
                        # Do it by batch if max_size_limit < size of the files
                        if int(sample_config['size']) <= int(self.max_size_limit):
                            branches[branch_id] = (sample_name, sample_id, period, das_dataset, sampleType, input_file_remote, alder_remote)
                            branch_id += 1
                        else:
                            split_nb = 0
                            while len(sample_config['files'])>0 :
                                filename_list = []
                                alder_list = []
                                index_pop = 0
                                sum_size = 0
                                for index, input_file_entry in enumerate(sample_config['files']):
                                    sum_size += input_file_entry['size']
                                    if (sum_size > int(self.max_size_limit)):
                                        if len(filename_list) == 0:
                                            raise RuntimeError(f'Max size too low: must be above the size of the biggest file in {sample_name}')
                                        branches[branch_id] = (sample_name + '_split-' +str(split_nb), sample_id, period, das_dataset, sampleType, filename_list, alder_list)
                                        branch_id += 1
                                        split_nb += 1
                                        break
                                    index_pop = index
                                    filename_list.append(input_file_entry['name'])
                                    alder_list.append(input_file_entry.get('adler32', None))
                                    #filling the leftover files --> it can be optimized to have smart file attribution to minimize the number of batches 
                                    if (index+1 == len(sample_config['files'])):
                                        branches[branch_id] = (sample_name + '_split-' +str(split_nb), sample_id, period, das_dataset, sampleType, filename_list, alder_list)
                                        branch_id += 1
                                        split_nb += 1
                                        break
                                
                                for popindex in range(index_pop+1):
                                    sample_config['files'].pop(0)
                sample_id += 1
        if len(missing_samples) > 0:
            self.publish_message("Missing samples: {}".format(', '.join(missing_samples)))
            if not self.ignore_missing_samples:
                raise RuntimeError("Missing samples has been detected.")
        
        self._cache_branches = True
        return branches

    # output path for the skimmed root file
    def output(self):
        if self.branch_data is None:
            #path to a file that should not exist
            return law.LocalFileTarget('thisfileshouldnotexist.tmp')

        sample_name_raw, sample_id, period, das_dataset, sampleType, input_file_remotes, alder_list = self.branch_data
        if bool(re.search('_split.*', sample_name_raw)):
            sample_name = re.sub('_split.*', '', sample_name_raw)
            split_number = int(re.sub(sample_name+'_split-', '', sample_name_raw))
        else:
            sample_name = sample_name_raw
            split_number = None

        if split_number is None :
            sample_out = os.path.join(self.central_path(), self.version, self.dataset_tier, period, sampleType, sample_name + '.root' )
        else:
            #if more than one file per batch, create a directory with all the split file from the same data sample in it
            sample_out = os.path.join(self.central_path(), self.version, self.dataset_tier, period, sampleType, sample_name , sample_name_raw + '.root' )

        return law.LocalFileTarget(sample_out)

    def run(self):
        sample_name_raw, sample_id, period, das_dataset, sampleType, input_file_remotes, alder_list = self.branch_data
        #load the skimming file
        skim_config_path = os.path.join(self.ana_path(), 'config', 'skims.yaml')
        with open(skim_config_path, 'r') as f:
            skim_config = yaml.safe_load(f)
        exclude_columns = ','.join(skim_config['common']['exclude_columns'])
        output_files = []
        selection = skim_config['lepselection']['selection']
        sampleType_selection = skim_config['lepselection']['sampleType']
        os.makedirs(self.local_central_path(), exist_ok=True)
        for n, input_file_remote in enumerate(input_file_remotes):
            adler32 = alder_list[n]
            if adler32 is not None:
                adler32 = int(adler32, 16)
            
            input_file_local = self.local_central_path(f'{sample_name_raw}_{n}_in.root')
            output_file = self.local_central_path(f'{sample_name_raw}_{n}_out.root')
            # First copy files locally 
            xrd_copy(input_file_remote, input_file_local, expected_adler32sum=adler32, silent=False)
            # Then skim each file individually
            skim_tree_cmd = [ 'python3', os.path.join(os.environ["ANALYSIS_PATH"], "scripts", "skim_tree.py"), '--input', input_file_local, '--output', output_file, '--input-tree', 'Events',
                             '--other-trees', 'LuminosityBlocks,Runs', '--exclude-columns', exclude_columns, '--verbose', '1' ]
            if sampleType in sampleType_selection:
                skim_tree_cmd.extend(['--sel', selection])
            sh_call(skim_tree_cmd, verbose=1)
            os.remove(input_file_local)
            output_files.append(output_file)
        output_path = self.output().path
        os.makedirs(self.output().dirname, exist_ok=True)
        if len(output_files) > 1:
            output_tmp = output_path + '.tmp.root'
            #Finaly, hadd the skimmed root files together
            sh_call(['python3', os.path.join(os.environ["ANALYSIS_PATH"], "scripts", "haddnano.py"), output_tmp] + output_files, verbose=1)
            for f in output_files:
                os.remove(f)
        elif len(output_files) == 1:
            output_tmp = output_files[0]
        else:
            raise RuntimeError("No output files found.")
        shutil.move(output_tmp, output_path)