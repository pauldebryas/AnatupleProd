import law
import luigi
import os
import re
from run_tools.law_customizations import Task
import yaml
import json
from run_tools.sh_tools import sh_call

class CreateVomsProxy(law.Task):
    time_limit = luigi.Parameter(default='24')

    def __init__(self, *args, **kwargs):
        super(CreateVomsProxy, self).__init__(*args, **kwargs)
        self.proxy_path = os.getenv("X509_USER_PROXY")
        if os.path.exists(self.proxy_path):
            proxy_info = self.get_proxy_info()
            timeleft = self.get_proxy_timeleft(proxy_info)
            if timeleft < float(self.time_limit):
                self.publish_message(f"Removing old proxy which expires in a less than {timeleft:.1f} hours.")
                os.remove(self.proxy_path)

    def output(self):
        return law.LocalFileTarget(self.proxy_path)

    def create_proxy(self, proxy_file):
        self.publish_message("Creating voms proxy...")
        proxy_file.makedirs()
        sh_call(['voms-proxy-init', '-voms', 'cms', '-rfc', '-valid', '192:00', '--out', proxy_file.path])

    def get_proxy_info(self):
        _, output = sh_call(['voms-proxy-info'], catch_stdout=True, split='\n')
        info = {}
        for line in output:
            match = re.match(r'^(.+) : (.+)', line)
            key = match.group(1).strip()
            info[key] = match.group(2)
        return info

    def get_proxy_timeleft(self, proxy_info):
        h,m,s = proxy_info['timeleft'].split(':')
        return float(h) + ( float(m) + float(s) / 60. ) / 60.

    def run(self):
        proxy_file = self.output()
        self.create_proxy(proxy_file)
        if not proxy_file.exists():
            raise RuntimeError("Unable to create voms proxy")

# Create yaml file with the sample list that have been processed for a specific year.
class CreateSamplesConfigFile(Task):
    #adding NanoAOD path of the samples in the config file 
    #AddNanoAODInfo = luigi.Parameter(default='False')
    AddNanoAODInfo = luigi.BoolParameter(default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    # relative path of the directory where the skimmed sample list is stored
    inputdir_path = luigi.Parameter(default='/afs/cern.ch/user/p/pdebryas/HNL_analysis/NanoProd/NanoProd/crab/')

    # requires VomsProxy to read files in DAS (if AddNanoAODInfo is chosen)
    def workflow_requires(self):
        return { "proxy": CreateVomsProxy.req(self) }
    
    def requires(self):
        return self.workflow_requires()

    def __init__(self, *args, **kwargs):
        super(CreateSamplesConfigFile, self).__init__(*args, **kwargs)
        #relative path where yaml file will be stored
        self.SamplesConfigFile_path = os.path.join(os.getenv("ANALYSIS_PATH"), 'config', f'samples_{self.periods}.yaml')
        #relative path where sample list is stored
        self.inputfiles_path = os.path.join(self.inputdir_path, f'Run2_{self.periods}') 

    def output(self):
        return law.LocalFileTarget(self.SamplesConfigFile_path)

    def get_str_prodMiniAOD(self):
        if self.periods == '2018': 
            return ['RunIISummer20UL18', 'MiniAOD']
        if self.periods == '2017':
            return ['RunIISummer20UL17', 'MiniAOD']
        if self.periods == '2016':
            return ['RunIISummer20UL16', 'MiniAOD']
        if self.periods == '2016_HIPM':
            return ['RunIISummer20UL16', 'MiniAODAPV']
        raise ValueError('Invalid year name: can be 2018, 2017, 2016 or 2016_HIPM')

    def get_nanoAOD_from_miniAOD(self, miniAOD, sample_type):
        if sample_type == 'data':
            splitminiAOD = miniAOD.split('MiniAOD')
            startOfquery= splitminiAOD[0]+ 'MiniAODv2_NanoAOD'
            query = startOfquery +'*/NANOAOD'
        else:  
            str_prodMiniAOD = self.get_str_prodMiniAOD()
            splitminiAOD = miniAOD.split(str_prodMiniAOD[0])
            startOfquery= splitminiAOD[0]+ str_prodMiniAOD[0] 
            endOfquery = splitminiAOD[1]
            endOfquery = endOfquery.replace('/MINIAODSIM', '')
            endOfquery = endOfquery.replace(str_prodMiniAOD[1], '')
            if self.periods == '2016_HIPM': 
                endOfquery = endOfquery[2:-4]
            else:
                endOfquery = endOfquery[2:-2]
            query = startOfquery +'*'+ endOfquery +'*'+ '/NANOAODSIM'
        # use the following command to look at the files corresonding to the data sample: dasgoclient -json -query 'dataset= ...' 
        _, output = sh_call(['dasgoclient', '-json', '-query', f'dataset={query}'], catch_stdout=True)
        output=json.loads(output)
        nanoAOD = []
        for dataset in output[0]['dataset']:
            nanoAOD.append(dataset['name'])
        nNanoAODfound = len(nanoAOD) 
        if nNanoAODfound == 0:
            print(f'Warning: For dataset {splitminiAOD[0]} no NanoAOD correspondance found (DAS query: {query})')
        if nNanoAODfound == 1:
            nanoAOD = nanoAOD[0]
        if nNanoAODfound > 1:
            print(f'Warning: For dataset {splitminiAOD[0]} multiple NanoAOD correspondance found (DAS query: {query})')
        return nanoAOD

    def run(self):
        output_samples_dict = {}
        for input_yamlfile in os.listdir(self.inputfiles_path):
            if input_yamlfile[-5:] != '.yaml':
                continue

            print(f'Reading {input_yamlfile} file...')

            with open(os.path.join(self.inputfiles_path, input_yamlfile), 'r') as f:
                samples = yaml.safe_load(f)

            if samples['config'] is None:
                raise ValueError('Missing config key in sample file')
            sample_list = list(samples.keys())
            sample_list.remove('config')

            if samples['config']['params']['sampleType'] is None:
                raise ValueError(f'Missing sampleType in sample file')
            
            sample_type = samples['config']['params']['sampleType']

            for sample in sample_list:
                print(f' - processing {sample}')
                output_samples_dict[sample] = {}
                output_samples_dict[sample]['sampleType'] = sample_type

                if 'ignoreFiles' in samples[sample]:
                    output_samples_dict[sample]['miniAOD'] = samples[sample]['inputDataset'] 
                    output_samples_dict[sample]['miniAOD_ignoreFiles'] = samples[sample]['ignoreFiles'] 
                else:
                    output_samples_dict[sample]['miniAOD'] = samples[sample]

                if sample_type == 'mc':
                    output_samples_dict[sample]['PhysicsType'] = input_yamlfile[:-5]
                    output_samples_dict[sample]['crossSection'] = sample
                if 'HNL' in input_yamlfile:
                    output_samples_dict[sample]['crossSection'] = 'HNL_tau_normalised'
                    output_samples_dict[sample]['mass'] = int(sample[sample.rfind("-") + 1:])
                    output_samples_dict[sample]['Vtau'] = 0.01
                if self.AddNanoAODInfo:
                    nanoAOD = self.get_nanoAOD_from_miniAOD(output_samples_dict[sample]['miniAOD'], output_samples_dict[sample]['sampleType'])
                    output_samples_dict[sample]['nanoAOD'] = nanoAOD
        
        with open(self.SamplesConfigFile_path, 'w') as f:
            yaml.safe_dump(output_samples_dict, f)
