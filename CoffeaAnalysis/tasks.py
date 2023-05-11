#!/usr/bin/env python
import law
import luigi
import pickle
import os
import importlib
import shutil

from coffea import processor
from coffea.nanoevents import NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = True

from CoffeaAnalysis.HNLAnalysis.CountEvents import CountEvents
from CoffeaAnalysis.task_helpers import files_from_path, cleanup_ds
from run_tools.law_customizations import Task, HTCondorWorkflow

# Base task inherited by all other classes (all parameters in there)
class BaseTaskAnalysis(Task):
    # tag that will be added to store pkl and anatuple files
    tag = luigi.Parameter(default='TEST')

    def load_samples(self, channel):
        # read config/samples_*.yalm input file and store useful information/branches
        self.load_sample_configs()
        self.load_xsecs()

        branch_index = 0
        branches = {}
        data_samples_list = {}
        for period, samples in self.samples.items():
            for sample_name in sorted(samples.keys()):
                sampleType = samples[sample_name].get('sampleType', None)
                crossSection = samples[sample_name].get('crossSection', None)
                exclude_samples = self.global_sample_params[period].get('exclude_samples', [])

                if exclude_samples is None or len(exclude_samples) == 0:
                    self.publish_message("Warning: Missing exclude_samples in samples_2018.yaml")
                if sampleType is None or len(sampleType) == 0:
                    self.publish_message("Missing sampleType for sample: {}".format(sample_name))
                    raise RuntimeError("Missing sampleType has been detected.")
                if sample_name in exclude_samples:
                    continue

                path_to_sample = os.path.join(os.path.join(self.central_path(), self.version, 'Run2_2018', sample_name))
                if os.path.exists(path_to_sample):
                    files = files_from_path(path_to_sample)
                else:
                    self.publish_message("Missing sample path for sample: {}".format(path_to_sample))
                    raise RuntimeError("Non existing sample path has been detected.")
                
                if sampleType != 'data':
                    if crossSection is None or len(crossSection) == 0:
                        self.publish_message("Missing crossSection for sample: {}".format(sample_name))
                        raise RuntimeError("Missing crossSection has been detected.")
                    else:
                        if type(self.xsecs[crossSection])== str:
                            self.publish_message("Warning: crossSection format for sample: {}".format(sample_name))
                            xsecs = eval(self.xsecs[crossSection])
                        else:
                            xsecs = self.xsecs[crossSection]
        
                    output_file = os.path.join(self.ana_path(), 'CoffeaAnalysis', 'results', self.tag+'_'+channel, 'cutflow_pkl', sample_name+str('_cutflow.pkl') )
                    branches[branch_index] = (sample_name, period, xsecs, sampleType, files, output_file)
                    branch_index += 1
                else:
                    data_samples_list[sample_name] = files_from_path(path_to_sample)

        return data_samples_list , branches, period


class RunCounter(BaseTaskAnalysis, HTCondorWorkflow, law.LocalWorkflow):

    def create_branch_map(self):
        path = os.path.join(self.ana_path(), 'CoffeaAnalysis', 'results')
        branches = {}
        branches[0] = path + '/counter.pkl'
        return branches
    
    def output(self):
        output_path = self.branch_data
        return self.local_analysis_target(output_path)

    def run(self):
        output_path = self.branch_data

        self.load_sample_configs()
        samples_list = {}
        for period, samples in self.samples.items():
            for sample_name in sorted(samples.keys()):
                sampleType = samples[sample_name].get('sampleType', None)
                exclude_samples = self.global_sample_params[period].get('exclude_samples', [])
                if exclude_samples is None or len(exclude_samples) == 0:
                    self.publish_message("Warning: Missing exclude_samples in samples_2018.yaml")
                if sampleType is None or len(sampleType) == 0:
                    self.publish_message("Missing sampleType for sample: {}".format(sample_name))
                    raise RuntimeError("Missing sampleType has been detected.")
                if sample_name in exclude_samples:
                    continue
                path_to_sample = os.path.join(os.path.join(self.central_path(), self.version, 'Run2_2018', sample_name))
                if os.path.exists(path_to_sample):
                    if sampleType != 'data':
                        samples_list[sample_name] = files_from_path(path_to_sample)
                else:
                    self.publish_message("Missing sample path for sample: {}".format(path_to_sample))
                    raise RuntimeError("Non existing sample path has been detected.")
        
        stitched_list = self.global_sample_params[period].get('stitched_list', [])
        if stitched_list is None or len(stitched_list) == 0:
            raise RuntimeError("Missing stitched_list in samples_2018.yaml")

        Drell_Yann_samples = stitched_list['DY_samples']
        WJets_samples= stitched_list['WJets_samples']
        Backgrounds_stitched = Drell_Yann_samples + WJets_samples

        event_counter_NotSelected = processor.run_uproot_job(
            samples_list,
            'EventsNotSelected',
            CountEvents(Backgrounds_stitched),
            processor.iterative_executor,
            {"schema": NanoAODSchema, 'workers': 6},
        )

        event_counter_Selected = processor.run_uproot_job(
            samples_list,
            'Events',
            CountEvents(Backgrounds_stitched),
            processor.iterative_executor,
            {"schema": NanoAODSchema, 'workers': 6},
        )

        event_counter = {}
        event_counter['sumw'] = {}
        event_counter['sumw_PUcorr'] = {}

        for sample in list(samples_list):
            event_counter['sumw'][sample] = event_counter_NotSelected['sumw'][sample] + event_counter_Selected['sumw'][sample]
            event_counter['sumw_PUcorr'][sample] = event_counter_NotSelected['sumw_PUcorr'][sample] + event_counter_Selected['sumw_PUcorr'][sample]

        self.output().dump(event_counter)


class Analysis(BaseTaskAnalysis, HTCondorWorkflow, law.LocalWorkflow):

    channel = luigi.Parameter(default='ttm')

    # requires RunCounter for scaling
    def workflow_requires(self):
        return { "Counter": RunCounter.req(self, branch=0) }

    def requires(self):
        return self.workflow_requires()

    def create_branch_map(self):

        #adding data to the branches
        if self.channel not in ['ttm','tmm','tem', 'tee', 'ttt','tte']:
            raise RuntimeError(f"Incorrect channel name: {self.channel}")

        if self.channel in ['ttm','tmm','tem']:
            self.dataHLT='SingleMuon'
        if self.channel in ['tee']:
            self.dataHLT='EGamma'
        if self.channel in ['ttt','tte']:
            self.dataHLT='Tau'

        # load MC branches 
        data_samples_list , branches, period = self.load_samples(self.channel)
        branch_index = len(branches)

        data_files = []
        data_sample_name = []
        for data_sample in data_samples_list.keys():
            if self.dataHLT in data_sample:
                data_files.append(data_samples_list[data_sample])
                data_sample_name.append(data_sample)

        output_file = os.path.join(self.ana_path(), 'CoffeaAnalysis', 'results', self.tag+'_'+self.channel, 'cutflow_pkl', f'{self.dataHLT}_{period}_cutflow.pkl')
        branches[branch_index] = (data_sample_name, period, 1., 'data', data_files, output_file)
        return branches
    
    def output(self):
        sample_name, period, xsecs, sampleType, files, output_file = self.branch_data
        return self.local_analysis_target(output_file)

    def run(self):

        self.load_sample_configs()
        tag = self.tag+'_'+self.channel
        sample_name, period, xsecs, sampleType, files, output_file = self.branch_data

        samples_list = {}
        if sampleType != 'data':
            samples_list[sample_name] = files
        else:
            i=0
            for sample in sample_name:
                samples_list[sample] = files[i]
                i = i+1
            sample_name = f'{sample[0:-1]}'

        output_root_folder = os.path.join(self.ana_path(), 'CoffeaAnalysis', 'results', tag, 'anatuple')
        if not os.path.exists(output_root_folder):
            os.makedirs(output_root_folder)

        output_pkl_folder = os.path.join(self.ana_path(), 'CoffeaAnalysis', 'results', tag,'cutflow_pkl')
        if not os.path.exists(output_pkl_folder):
            os.makedirs(output_pkl_folder)

        output_tmp_folder = f'/afs/cern.ch/work/p/pdebryas/HNL/tmp/{tag}/{sample_name}/'
        if os.path.exists(output_tmp_folder):
            print(f'A tmp folder which store tmp anatuple files exist already for dataset {sample_name}: being deleted')
            shutil.rmtree(output_tmp_folder)

        stitched_list = self.global_sample_params[period].get('stitched_list', [])
        if stitched_list is None or len(stitched_list) == 0:
            raise RuntimeError("Missing stitched_list in samples_2018.yaml")

        module = importlib.import_module(f'CoffeaAnalysis.HNLAnalysis.HNLAnalysis_{self.channel}')
        HNLAnalysis = getattr(module, f'HNLAnalysis_{self.channel}')

        result = processor.run_uproot_job(
            samples_list,
            "Events",
            HNLAnalysis(stitched_list, tag, xsecs),
            processor.iterative_executor,
            {"schema": NanoAODSchema, 'workers': 6},
        )

        self.output().dump(result)

        cleanup_ds(sample_name, output_tmp_folder, output_root_folder)