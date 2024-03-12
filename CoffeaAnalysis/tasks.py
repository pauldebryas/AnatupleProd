#!/usr/bin/env python
import law
import luigi
import os
import importlib
import shutil

from coffea import processor
from coffea.nanoevents import NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False

from CoffeaAnalysis.HNLAnalysis.CountEvents import CountEvents
from CoffeaAnalysis.task_helpers import files_from_path, cleanup_ds
from run_tools.law_customizations import Task, HTCondorWorkflow


class RunCounter(Task, HTCondorWorkflow, law.LocalWorkflow):

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
        
class Analysis(Task, HTCondorWorkflow, law.LocalWorkflow):

    tag = luigi.Parameter(default='TEST')
    channel = luigi.Parameter(default='ttm')

    # requires RunCounter for scaling
    def workflow_requires(self):
        return { "Counter": RunCounter.req(self, branch=0) }

    def requires(self):
        return self.workflow_requires()
    
    def load_samples(self):
        # read config/samples_*.yalm input file and store useful information/branches
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

                path_to_sample = os.path.join(os.path.join(self.central_path_nanoAOD(), f'Run2_{period}', sample_name))
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
        
                    output_file = os.path.join(self.output_anatuple(), self.tag, self.channel, 'cutflow_pkl', sample_name+str('_cutflow.pkl') )
                    MC_branches[branch_index] = (sample_name, get_Xsec, sampleType, files, output_file)
                    branch_index += 1
                else:
                    data_samples_list[sample_name] = files_from_path(path_to_sample)

        return data_samples_list , MC_branches
    
    def create_branch_map(self):

        #adding data to the branches
        if self.channel not in ['ttm','tmm','tem', 'tee', 'ttt','tte']:
            raise RuntimeError(f"Incorrect channel name: {self.channel}")

        if self.channel in ['ttm','tmm','tem']:
            self.dataHLT='SingleMuon'
        if self.channel in ['tee', 'tte']:
            self.dataHLT='EGamma'
        #if self.channel in ['ttt','tte']:
        #    self.dataHLT='Tau'
        #    self.dataHLT='EGamma'

        # load MC branches 
        data_samples_list , branches = self.load_samples()
        branch_index = len(branches)

        data_files = []
        data_sample_name = []
        for data_sample in data_samples_list.keys():
            if self.dataHLT in data_sample:
                data_files.append(data_samples_list[data_sample])
                data_sample_name.append(data_sample)

        output_file = os.path.join(self.output_anatuple(), self.tag, self.channel, 'cutflow_pkl', f'{self.dataHLT}_{self.periods}_cutflow.pkl')
        branches[branch_index] = (data_sample_name, None, 'data', data_files, output_file)
        return branches
    
    def output(self):
        sample_name, xsecs, sampleType, files, output_file = self.branch_data
        return self.local_analysis_target(output_file)

    def run(self):

        #self.load_sample_configs()
        sample_name, xsecs, sampleType, files, output_file = self.branch_data

        samples_list = {}
        if sampleType != 'data':
            samples_list[sample_name] = files
        else:
            i=0
            for sample in sample_name:
                samples_list[sample] = files[i]
                i = i+1
            sample_name = f'{sample[0:-1]}'

        output_root_folder = os.path.join(self.output_anatuple(), self.tag, self.channel, 'anatuple')
        if not os.path.exists(output_root_folder):
            os.makedirs(output_root_folder)

        output_pkl_folder = os.path.join(self.output_anatuple(), self.tag, self.channel, 'cutflow_pkl')
        if not os.path.exists(output_pkl_folder):
            os.makedirs(output_pkl_folder)

        output_tmp_folder = f'/afs/cern.ch/work/p/pdebryas/HNL/tmp/{self.tag}/{self.channel}/{sample_name}/'
        if os.path.exists(output_tmp_folder):
            print(f'A tmp folder which store tmp anatuple files exist already for dataset {sample_name}: being deleted')
            shutil.rmtree(output_tmp_folder)

        self.load_global_params()
        stitched_list = self.stitched_samples[self.periods]
        if stitched_list is None or len(stitched_list) == 0:
            raise RuntimeError(f"Missing stitched_list in samples_{self.periods}.yaml")

        module = importlib.import_module(f'CoffeaAnalysis.HNLAnalysis.HNLAnalysis_{self.channel}')
        HNLAnalysis = getattr(module, f'HNLAnalysis_{self.channel}')

        result = processor.run_uproot_job(
            samples_list,
            "Events",
            HNLAnalysis(stitched_list, self.tag, xsecs, self.periods),
            processor.iterative_executor,
            {"schema": NanoAODSchema, 'workers': 6},
        )

        self.output().dump(result)

        cleanup_ds(sample_name, output_tmp_folder, output_root_folder)