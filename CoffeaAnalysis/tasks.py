#!/usr/bin/env python
import law
import luigi
import pickle
import os
import importlib

from coffea import processor
from coffea.nanoevents import NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = True

from CoffeaAnalysis.HNLAnalysis.CountEvents import CountEvents, CountEvents_stitched_samples
from CoffeaAnalysis.HNLAnalysis.helpers import files_from_path
from run_tools.law_customizations import Task, HTCondorWorkflow

from CoffeaAnalysis.HNLAnalysis.helpers import import_stitching_weights

# Base task inherited by all other classes (all parameters in there)
class BaseTaskAnalysis(Task):
    # tag that will be added to produced pkl files and not mixt up with different analysis
    tag = luigi.Parameter(default='TEST')

class RunCounter(BaseTaskAnalysis, HTCondorWorkflow, law.LocalWorkflow):

    def create_branch_map(self):
        path = os.path.join(self.ana_path(), 'CoffeaAnalysis', 'results', self.tag)
        branches = {}
        branches[0] = path + '/counter.pkl'
        return branches
    
    def output(self):
        output_path = self.branch_data
        return self.local_analysis_target(output_path)

    def run(self):
        output_path = self.branch_data
        tag = self.tag

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

        samples_stitched = {}
        for element in Backgrounds_stitched:
            samples_stitched[element] = samples_list[element]

        event_counter_NotSelected = processor.run_uproot_job(
            samples_stitched,
            'EventsNotSelected',
            CountEvents_stitched_samples(),
            processor.iterative_executor,
            {"schema": NanoAODSchema, 'workers': 6},
        )

        event_counter_Selected = processor.run_uproot_job(
            samples_stitched,
            'Events',
            CountEvents_stitched_samples(),
            processor.iterative_executor,
            {"schema": NanoAODSchema, 'workers': 6},
        )

        event_counter = processor.run_uproot_job(
            samples_list,
            'Runs',
            CountEvents(),
            processor.iterative_executor,
            {"schema": NanoAODSchema, 'workers': 6},
        )

        for sample in list(samples_stitched):
            event_counter['sumw'][sample] = event_counter_NotSelected['sumw_fixed'][sample] + event_counter_Selected['sumw_fixed'][sample]

        self.output().dump(event_counter)

class RunAnalysis(BaseTaskAnalysis, HTCondorWorkflow, law.LocalWorkflow):

    # requires RunCounter for scaling
    def workflow_requires(self):
        return { "Counter": RunCounter.req(self) }
    
    def requires(self):
        return self.workflow_requires()

    def create_branch_map(self):
        path = os.path.join(self.ana_path(), 'CoffeaAnalysis', 'results', self.tag)
        output_files = [
            path + '/result_tee_regionA.pkl',
            path + '/result_tmm_regionA.pkl',
            path + '/result_tte_regionA.pkl',
            path + '/result_ttm_regionA.pkl',
            path + '/result_tem_SS_regionA.pkl',
            path + '/result_tem_OS_regionA.pkl',
            path + '/result_tee_regionB.pkl',
            path + '/result_tmm_regionB.pkl',
            path + '/result_tte_regionB.pkl',
            path + '/result_ttm_regionB.pkl',
            path + '/result_tem_SS_regionB.pkl',
            path + '/result_tem_OS_regionB.pkl',
            path + '/result_tee_regionC.pkl',
            path + '/result_tmm_regionC.pkl',
            path + '/result_tte_regionC.pkl',
            path + '/result_ttm_regionC.pkl',
            path + '/result_tem_SS_regionC.pkl',
            path + '/result_tem_OS_regionC.pkl',
            path + '/result_tee_regionD.pkl',
            path + '/result_tmm_regionD.pkl',
            path + '/result_tte_regionD.pkl', 
            path + '/result_ttm_regionD.pkl',
            path + '/result_tem_SS_regionD.pkl',
            path + '/result_tem_OS_regionD.pkl']
        branches = {}
        for i in range(len(output_files)):
            branches[i] = output_files[i]
        return branches
    
    def output(self):
        output_path = self.branch_data
        # output 6x4 + 1 pkl files
        return self.local_analysis_target(output_path)

    def run(self):
        output_path = self.branch_data
        tag = self.tag

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
                    samples_list[sample_name] = files_from_path(path_to_sample)
                else:
                    self.publish_message("Missing sample path for sample: {}".format(path_to_sample))
                    raise RuntimeError("Non existing sample path has been detected.")
        
        if self.branch in [0,1,2,3,4,5]:
            region = 'A'
        if self.branch in [6,7,8,9,10,11]:
            region = 'B'
        if self.branch in [12,13,14,15,16,17]:
            region = 'C'
        if self.branch in [18,19,20,21,22,23]:
            region = 'D'

        if self.branch in [0,6,12,18]:
            channel = 'tee'
        if self.branch in [1,7,13,19]:
            channel = 'tmm'
        if self.branch in [2,8,14,20]:
            channel = 'tte'
        if self.branch in [3,9,15,21]:
            channel = 'ttm'
        if self.branch in [4,10,16,22]:
            channel = 'tem_SS'
        if self.branch in [5,11,17,23]:
            channel = 'tem_OS'

        if region not in ['A','B','C','D']:
            raise RuntimeError("Incorrect region name detected: {}".format(region))

        if channel not in ['tee','tmm','tte','ttm', 'tem_SS', 'tem_OS']:
            raise RuntimeError("Incorrect channel name detected: {}".format(channel))

        stitched_list = self.global_sample_params[period].get('stitched_list', [])
        if stitched_list is None or len(stitched_list) == 0:
            raise RuntimeError("Missing stitched_list in samples_2018.yaml")

        print(f'Running for region {region} and channel {channel}')
        module = importlib.import_module(f'CoffeaAnalysis.HNLAnalysis.HNLAnalysis_{channel}')
        HNLAnalysis = getattr(module, f'HNLAnalysis_{channel}')
        
        result = processor.run_uproot_job(
            samples_list,
            "Events",
            HNLAnalysis(region, stitched_list),
            processor.iterative_executor,
            {"schema": NanoAODSchema, 'workers': 6},
        )
        self.output().dump(result)