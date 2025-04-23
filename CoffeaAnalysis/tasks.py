#!/usr/bin/env python
import law
import luigi
import os
import importlib
import shutil
import warnings
import random
import pickle

#from coffea import dataset_tools
from coffea.nanoevents import NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False
warnings.simplefilter(action='ignore', category=FutureWarning)

from CoffeaAnalysis.HNLAnalysis.CountEvents import CountEvents
from CoffeaAnalysis.task_helpers import files_from_path, cleanup_ds, merge_pkl_files
from run_tools.law_customizations import Task, HTCondorWorkflow
from coffea import processor

class RunCounter(Task, HTCondorWorkflow, law.LocalWorkflow):
    debugMode = luigi.BoolParameter(default=False)

    def create_branch_map(self):
        path_to_file = os.path.join(self.output_anatuple() ,'counter.pkl')
        branches = {}
        branches[0] = path_to_file
        return branches
    
    def output(self):
        output_path = self.branch_data
        return self.local_analysis_target(output_path)

    def load_MCsamples(self):
        self.load_sample_configs()
        self.load_global_params()
        samples_list = {}
        for period, samples in self.samples.items():
            if self.excluded_samples[period] is None or len(self.excluded_samples[period]) == 0:
                self.publish_message(f"Warning: Missing exclude_samples in samples_{self.periods}.yaml")
            for sample_name in sorted(samples.keys()):
                sampleType = samples[sample_name].get('sampleType', None)
                if sampleType is None or len(sampleType) == 0:
                    self.publish_message("Missing sampleType for sample: {}".format(sample_name))
                    raise RuntimeError("Missing sampleType has been detected.")
                if sampleType == 'data':
                    continue
                if sample_name in self.excluded_samples[period]:
                    continue
                path_to_sample = os.path.join(os.path.join(self.central_path_nanoAOD(), f'Run2_{period}', sample_name))
                if os.path.exists(path_to_sample):
                    samples_list[sample_name] = files_from_path(path_to_sample)
                else:
                    self.publish_message("Missing sample path for sample: {}".format(path_to_sample))
                    raise RuntimeError("Non existing sample path has been detected.")
        return samples_list

    def load_StitchedSamples(self):
        self.load_sample_configs()
        if self.stitched_samples[self.periods] is None or len(self.stitched_samples[self.periods]) == 0:
            raise RuntimeError(f"Missing stitched_list in samples_{self.periods}.yaml")
        return self.stitched_samples[self.periods]

    def run(self):
        samples_list = self.load_MCsamples()
        Backgrounds_stitched = self.load_StitchedSamples()

        if self.debugMode:
            samples_list = random.sample(list(samples_list), 5)

        event_counter_NotSelected = processor.run_uproot_job(
            samples_list,
            'EventsNotSelected',
            CountEvents(Backgrounds_stitched, self.periods),
            processor.iterative_executor,
            {"schema": NanoAODSchema, 'workers': 6},
        )

        event_counter_Selected = processor.run_uproot_job(
            samples_list,
            'Events',
            CountEvents(Backgrounds_stitched, self.periods),
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

files_to_pop ={
    'TTToSemiLeptonic': []
}

class RunAnalysis(Task, HTCondorWorkflow, law.LocalWorkflow):

    tag = luigi.Parameter(default='TEST')
    channel = luigi.Parameter(default='ttm')
    debugMode = luigi.BoolParameter(default=False)

    # requires RunCounter for scaling
    def workflow_requires(self):
        return { "Counter": RunCounter.req(self, branch=0) }

    def requires(self):
        return self.workflow_requires()
    
    def load_dataHLT(self):
        #adding data to the branches
        if self.channel not in ['ttm','tmm','tem', 'tee', 'ttt','tte','tte_DiTau']:
            raise RuntimeError(f"Incorrect channel name: {self.channel}")

        if self.channel in ['ttm','tmm','tem']:
            self.dataHLT='SingleMuon'
        if self.channel in ['tee', 'tte']:
            if self.periods == '2018':
                self.dataHLT='EGamma'
            else:
                self.dataHLT='SingleElectron'
        return
    
    def create_branch_map(self):
        if not os.path.exists(os.path.join(self.output_anatuple(), self.tag, 'tmp', self.channel, 'anatuple')):
            os.makedirs(os.path.join(self.output_anatuple(), self.tag, 'tmp', self.channel, 'anatuple'))
        self.load_dataHLT()
        # load MC branches 
        data_samples_list , branches = self.load_samples(files_to_pop)
        branch_index = len(branches)
        for data_sample in data_samples_list.keys():
            if self.dataHLT in data_sample:
                output_file = os.path.join(self.output_anatuple(), self.tag, 'tmp', self.channel, 'anatuple', f'{data_sample}_anatuple.root')
                branches[branch_index] = (data_sample, None, 'data', data_samples_list[data_sample], output_file, self.dataHLT)
                branch_index = branch_index+1
        #print(branches)
        return branches
    
    def output(self):
        name, xsecs, sampleType, files, output_file, SampleName = self.branch_data
        return self.local_analysis_target(output_file)

    def run(self):
        self.load_dataHLT()
        name, xsecs, sampleType, files, output_file, SampleName = self.branch_data

        samples_list = {}
        if sampleType != 'data':
            samples_list[SampleName] = files
        else:
            samples_list[name] = files


        if name == SampleName:
            output_pkl_folder = os.path.join(self.output_anatuple(), self.tag, self.channel, 'cutflow_pkl')
            output_root_folder = os.path.join(self.output_anatuple(), self.tag, self.channel, 'anatuple')
        else:
            output_pkl_folder = os.path.join(self.output_anatuple(), self.tag, 'tmp', self.channel, 'cutflow_pkl')
            output_root_folder = os.path.join(self.output_anatuple(), self.tag, 'tmp', self.channel, 'anatuple')

        if not os.path.exists(output_pkl_folder):
            os.makedirs(output_pkl_folder)
        if not os.path.exists(output_root_folder):
            os.makedirs(output_root_folder)

        output_tmp_folder = f'/afs/cern.ch/work/p/pdebryas/HNL/tmp/{self.periods}/{self.tag}/{self.channel}/{name}/'
        if os.path.exists(output_tmp_folder):
            print(f'A tmp folder which store tmp anatuple files exist already for dataset {name}: being deleted')
            shutil.rmtree(output_tmp_folder)

        self.load_global_params()
        stitched_list = self.stitched_samples[self.periods]
        if stitched_list is None or len(stitched_list) == 0:
            raise RuntimeError(f"Missing stitched_list in samples_{self.periods}.yaml")

        module = importlib.import_module(f'CoffeaAnalysis.HNLAnalysis.channels.HNLAnalysis_{self.channel}')
        HNLAnalysis = getattr(module, f'HNLAnalysis_{self.channel}')
        my_processor = HNLAnalysis(stitched_list, self.tag, xsecs, self.periods, self.dataHLT, self.debugMode, name)

        result = processor.run_uproot_job(
            samples_list,
            "Events",
            my_processor,
            processor.iterative_executor,
            {"schema": NanoAODSchema, 'workers': 6},
        )

        # Save the result as a .pkl file
        with open(os.path.join(output_pkl_folder,f'{name}_cutflow.pkl'), "wb") as f:
            pickle.dump(result, f)

        cleanup_ds(name, output_tmp_folder, output_root_folder)

class RunPostProcess(Task, HTCondorWorkflow, law.LocalWorkflow):

    tag = luigi.Parameter(default='TEST')
    channel = luigi.Parameter(default='ttm')

    # requires Analysis
    def workflow_requires(self):
        return { "Analysis": RunAnalysis.req(self) }

    def requires(self):
        return self.workflow_requires()
        
    def load_dataHLT(self):
        #adding data to the branches
        if self.channel not in ['ttm','tmm','tem', 'tee', 'ttt','tte','tte_DiTau']:
            raise RuntimeError(f"Incorrect channel name: {self.channel}")

        if self.channel in ['ttm','tmm','tem']:
            self.dataHLT='SingleMuon'
        if self.channel in ['tee', 'tte']:
            if self.periods == '2018':
                self.dataHLT='EGamma'
            else:
                self.dataHLT='SingleElectron'
        return

    def create_branch_map(self):
        # load MC branches 
        data_samples_list , branches_RunAnalysis = self.load_samples(files_to_pop)
        self.load_dataHLT()

        merged_files_list = {}
        for i in range(len(branches_RunAnalysis)):
            (name, get_Xsec, sampleType, files, output_file, sample_name) = branches_RunAnalysis[i]
            if sample_name not in merged_files_list.keys():
                merged_files_list[sample_name] = [output_file]
            else:
                merged_files_list[sample_name].append(output_file)

        output_file_data = []
        for data_sample in data_samples_list.keys():
            if self.dataHLT in data_sample:
                output_file_data.append(os.path.join(self.output_anatuple(), self.tag, 'tmp',self.channel, 'anatuple', f'{data_sample}_anatuple.root'))
        merged_files_list[self.dataHLT] = output_file_data

        branches = {}
        i = 0
        for sample_name, merged_files in merged_files_list.items():
            if sample_name == self.dataHLT:
                output_file = os.path.join(self.output_anatuple(), self.tag, self.channel, 'anatuple', f'{sample_name}_{self.periods}_anatuple.root')
            else:
                output_file = os.path.join(self.output_anatuple(), self.tag, self.channel, 'anatuple', f'{sample_name}_anatuple.root')
            branches[i] = (sample_name, merged_files, output_file)
            i = i+1

        #print(branches)
        return branches
    
    def output(self):
        sample_name, merged_files, output_file = self.branch_data
        return self.local_analysis_target(output_file)

    def run(self):
        sample_name, merged_files_root, output_file_root = self.branch_data

        merged_files_pkl = []
        for file in merged_files_root:
            output_pkl_folder = os.path.join(self.output_anatuple(), self.tag, self.channel, 'cutflow_pkl')
            tmp_pkl_folder = os.path.join(self.output_anatuple(), self.tag, 'tmp', self.channel, 'cutflow_pkl')
            filename = file.split('/')[-1]
            pkl_file = os.path.join(tmp_pkl_folder, filename.replace('_anatuple.root', '')+'_cutflow.pkl')
            merged_files_pkl.append(pkl_file)
        output_file_pkl = os.path.join(output_pkl_folder, f'{sample_name}_cutflow.pkl')

        # merge root files
        print(f"Merge {len(merged_files_root)} files into {output_file_root}")
        filelist_cmd = ''
        for file in merged_files_root:
            filelist_cmd = filelist_cmd + file + ' '
        hadd_cmd = 'hadd -n 11 ' + output_file_root + ' ' + filelist_cmd 
        print('Running the folowing command:')
        print(hadd_cmd)
        os.system(hadd_cmd)
        print('')

        #merge pkl file
        merge_pkl_files(merged_files_pkl, output_file_pkl)
        print('')
