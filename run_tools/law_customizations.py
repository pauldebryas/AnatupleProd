import law
import luigi
import math
import os
import yaml

law.contrib.load("htcondor")

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

    max_runtime = law.DurationParameter(default=12.0, unit="h", significant=False, description="maximum runtime, default unit is hours, default: 12")

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
        config.custom_content.append(("requirements", "(OpSysAndVer =?= \"CentOS7\")"))
        # maximum runtime
        config.custom_content.append(("+MaxRuntime", int(math.floor(self.max_runtime * 3600)) - 1))
        # copy the entire environment
        config.custom_content.append(("getenv", "true"))

        log_path = os.path.join(ana_data_path, "logs")
        os.makedirs(log_path, exist_ok=True)
        config.custom_content.append(("log", os.path.join(log_path, 'job.$(ClusterId).$(ProcId).log')))
        config.custom_content.append(("output", os.path.join(log_path, 'job.$(ClusterId).$(ProcId).out')))
        config.custom_content.append(("error", os.path.join(log_path, 'job.$(ClusterId).$(ProcId).err')))
        return config
