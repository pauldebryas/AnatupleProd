import law
import luigi
import math
import os
import yaml

law.contrib.load("htcondor")
from CoffeaAnalysis.task_helpers import files_from_path, get_size_in_gb, split_list

class Task(law.Task):
    """
    Base task that which will be inherited by all the other tasks.
    """
    periods = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super(Task, self).__init__(*args, **kwargs)
        self.all_periods = [ p for p in self.periods.split(',') if len(p) > 0 ]

    def load_sample_configs(self):
        self.samples = {}
        for period in self.all_periods:
            self.samples[period] = {}
            sample_config = os.path.join(self.ana_path(), 'config', f'samples_{period}.yaml')
            with open(sample_config, 'r') as f:
                samples = yaml.safe_load(f)
            for key, value in samples.items():
                if(type(value) != dict):
                    raise RuntimeError(f'Invalid sample definition period="{period}", sample_name="{key}"' )
                else:
                    self.samples[period][key] = value

    def load_global_params(self):
        self.stitched_samples = {}
        self.excluded_samples = {}
        for period in self.all_periods:
            self.stitched_samples[period] = {}
            self.excluded_samples[period] = {}
            sample_config = os.path.join(self.ana_path(), 'config', f'global_params_{period}.yaml')
            with open(sample_config, 'r') as f:
                samples = yaml.safe_load(f)
            for key, value in samples.items():
                if key == 'excluded_samples':
                    self.excluded_samples[period] = value
                if key == 'stitched_samples':
                    self.stitched_samples[period] = value

    def load_xsecs(self):
        self.xsecs = {}
        self.xsecs_unc = {}
        xsec_file = os.path.join(self.ana_path(), 'config', 'crossSections13TeV.yaml')
        with open(xsec_file, 'r') as f:
            xsec_load = yaml.safe_load(f)
        for sample, value in xsec_load.items():
            self.xsecs[sample] = value['crossSec']
            self.xsecs_unc[sample] = value['unc']

    def load_samples(self, files_to_pop = {}):
        MC_branches = {}
        data_samples_list = {}
    
        self.load_sample_configs()
        self.load_global_params()
        self.load_xsecs()
        branch_index = 0
        for period, samples in self.samples.items():
            for sample_name in sorted(samples.keys()):
                if sample_name in self.excluded_samples[period]:
                    continue
                sampleType = samples[sample_name].get('sampleType', None)
                if sampleType is None or len(sampleType) == 0:
                    self.publish_message("Missing sampleType for sample: {}".format(sample_name))
                    raise RuntimeError("Missing sampleType has been detected.")
                path_to_sample = os.path.join(os.path.join(self.central_path_nanoAOD(), f'Run2_{period}', samples[sample_name].get('HTTprodName')))
                size_gb = get_size_in_gb(path_to_sample)
                if os.path.exists(path_to_sample):
                    files = files_from_path(path_to_sample)
                else:
                    self.publish_message("Missing sample path for sample: {}".format(path_to_sample))
                    raise RuntimeError("Non existing sample path has been detected.")
                
                if sampleType != 'data':
                    crossSection = samples[sample_name].get('crossSection', None)
                    if crossSection is None or len(crossSection) == 0:
                        self.publish_message("Missing crossSection reference for sample: {}".format(sample_name))
                        raise RuntimeError("Missing crossSection has been detected.")
                    else:
                        get_Xsec = self.xsecs.get(crossSection, None)
                        if type(get_Xsec) == str:
                            #self.publish_message("Warning: crossSection format for sample: {}".format(sample_name))
                            get_Xsec = eval(get_Xsec)
                        if get_Xsec == None:
                            self.publish_message(f"Warning: crossSection in crossSections13TeV.yaml missing for sample: {sample_name} ")
        
                    if size_gb >= 200:
                        SplitInt = 20
                        subFiles = split_list(files, SplitInt)
                        for nSubFile in range(SplitInt):
                            files_split = subFiles[nSubFile]
                            output_file_split = os.path.join(self.output_anatuple(), self.tag, 'tmp',self.channel, 'anatuple', sample_name+str(f'_{nSubFile}_anatuple.root') )
                            if sample_name in files_to_pop.keys():
                                for file_to_pop in files_to_pop[sample_name]:
                                    if file_to_pop in files_split:
                                        files_split.remove(file_to_pop)
                                MC_branches[branch_index] = (sample_name+str(f'_{nSubFile}'), get_Xsec, sampleType, files_split, output_file_split, sample_name)
                            else:
                                MC_branches[branch_index] = (sample_name+str(f'_{nSubFile}'), get_Xsec, sampleType, files_split, output_file_split, sample_name)
                            branch_index += 1
                    else:
                        output_file = os.path.join(self.output_anatuple(), self.tag, self.channel, 'anatuple', sample_name+str('_anatuple.root') )
                        MC_branches[branch_index] = (sample_name, get_Xsec, sampleType, files, output_file, sample_name)
                        branch_index += 1
                else:
                    data_samples_list[sample_name] = files_from_path(path_to_sample)
        
        return data_samples_list , MC_branches
    
    def output_anatuple(self):
        output_anatuple_path = os.path.join(self.central_path_anatuple(),'anatuple' , self.periods)
        os.makedirs(output_anatuple_path, exist_ok=True)
        return output_anatuple_path
    
    def ana_path(self):
        return os.getenv("ANALYSIS_PATH")

    def ana_data_path(self):
        return os.getenv("ANALYSIS_DATA_PATH")

    def central_path_anatuple(self):
        return os.getenv("CENTRAL_STORAGE_ANATUPLE")

    def central_path_nanoAOD(self):
        return os.getenv("CENTRAL_STORAGE_NANOAOD")

    def local_analysis_path(self, *path):
        parts = (self.ana_path(),) + path
        return os.path.join(*parts)
    
    def local_data_path(self, *path):
        parts = (self.ana_data_path(),) + path
        return os.path.join(*parts)

    def local_central_path_anatuple(self, *path):
        parts = (self.central_path_anatuple(),) + path
        return os.path.join(*parts)

    def local_central_path_nanoAOD(self, *path):
        parts = (self.central_path_nanoAOD(),) + path
        return os.path.join(*parts)
    
    def local_analysis_target(self, *path):
        return law.LocalFileTarget(self.local_analysis_path(*path))
        
    def local_data_target(self, *path):
        return law.LocalFileTarget(self.local_data_path(*path))

    def local_central_target(self, *path):
        return law.LocalFileTarget(self.local_central_path(*path))

class HTCondorWorkflow(law.htcondor.HTCondorWorkflow):
    """
    Batch systems are typically very heterogeneous by design, and so is HTCondor. Law does not aim
    to "magically" adapt to all possible HTCondor setups which would certainly end in a mess.
    Therefore we have to configure the base HTCondor workflow in law.contrib.htcondor to work with
    the CERN HTCondor environment. In most cases, like in this example, only a minimal amount of
    configuration is required.
    """

    max_runtime = law.DurationParameter(
        default=24.0, 
        unit="h", 
        significant=False, 
        description="maximum runtime, default unit is hours, default: 12"
    )

    transfer_logs = luigi.BoolParameter(
        default=True,
        significant=False,
        description="transfer job logs to the output directory; default: True",
    )

    def htcondor_output_directory(self):
        # the directory where submission meta data should be stored
        return law.LocalDirectoryTarget(self.local_data_path())

    def htcondor_bootstrap_file(self):
        # each job can define a bootstrap file that is executed prior to the actual job
        # in order to setup software and environment variables
        return law.util.rel_path(__file__, "bootstrap.sh")

    def htcondor_job_config(self, config, job_num, branches):
        ana_path = os.getenv("ANALYSIS_PATH")
        ana_data_path = os.getenv("ANALYSIS_DATA_PATH")
        # render_variables are rendered into all files sent with a job
        config.render_variables["analysis_path"] = ana_path
        # force to run on CC7, http://batchdocs.web.cern.ch/batchdocs/local/submit.html#os-choice
        #config.custom_content.append(("requirements", "(OpSysAndVer =?= \"CentOS9\")"))
        # maximum runtime
        config.custom_content.append(("+MaxRuntime", int(math.floor(self.max_runtime * 3600)) - 1))
        # copy the entire environment
        config.custom_content.append(("getenv", "true"))

        log_path = os.path.join(ana_data_path, "logs")
        os.makedirs(log_path, exist_ok=True)
        config.custom_content.append(("log", os.path.join(log_path, f'job.$(ClusterId).{job_num}.log')))
        config.custom_content.append(("output", os.path.join(log_path, f'job.$(ClusterId).{job_num}.out')))
        config.custom_content.append(("error", os.path.join(log_path, f'job.$(ClusterId).{job_num}.err')))
        return config
