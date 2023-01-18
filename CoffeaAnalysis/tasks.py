#!/usr/bin/env python
import law
import luigi
import pickle
import os

from coffea import processor
from coffea.nanoevents import NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = True

from CoffeaAnalysis.HNLAnalysis.CountEvents import CountEvents, CountEvents_stitched_samples
from CoffeaAnalysis.HNLAnalysis.HNLAnalysis_tee import HNLAnalysis_tee
from CoffeaAnalysis.HNLAnalysis.HNLAnalysis_tmm import HNLAnalysis_tmm
from CoffeaAnalysis.HNLAnalysis.HNLAnalysis_tte import HNLAnalysis_tte
from CoffeaAnalysis.HNLAnalysis.HNLAnalysis_ttm import HNLAnalysis_ttm
from CoffeaAnalysis.HNLAnalysis.HNLAnalysis_tem_SS import HNLAnalysis_tem_SS
from CoffeaAnalysis.HNLAnalysis.HNLAnalysis_tem_OS import HNLAnalysis_tem_OS
from CoffeaAnalysis.HNLAnalysis.helpers import files_from_path

from run_tools.law_customizations import Task, HTCondorWorkflow

class RunAnalysis(Task, HTCondorWorkflow, law.LocalWorkflow):
    # tag that will be added to produced pkl files and not mixt up with different analysis
    tag = luigi.Parameter()

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
            path + '/result_tem_OS_regionD.pkl',
            path + '/counter.pkl']
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

        self.load_sample_analysis()
        samples_list = {}
        for period, samples in self.samples.items():
            for sample_name in sorted(samples.keys()):
                sampleType = samples[sample_name].get('sampleType', None)
                if sampleType is None or len(sampleType) == 0:
                    self.publish_message("Missing sampleType for sample: {}".format(sample_name))
                    raise RuntimeError("Missing sampleType has been detected.")
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

        if self.branch != 24:

            if region not in ['A','B','C','D']:
                raise RuntimeError("Incorrect region name detected: {}".format(region))

            if channel not in ['tee','tmm','tte','ttm', 'tem_SS', 'tem_OS']:
                raise RuntimeError("Incorrect channel name detected: {}".format(channel))

            if  channel == 'tee':
                result = processor.run_uproot_job(
                    samples_list,
                    "Events",
                    HNLAnalysis_tee(region),
                    processor.iterative_executor,
                    {"schema": NanoAODSchema, 'workers': 6},
                )
                self.output().dump(result)

            if  channel == 'tmm':
                result = processor.run_uproot_job(
                    samples_list,
                    "Events",
                    HNLAnalysis_tmm(region),
                    processor.iterative_executor,
                    {"schema": NanoAODSchema, 'workers': 6},
                )
                self.output().dump(result)

            if  channel == 'tte':
                result = processor.run_uproot_job(
                    samples_list,
                    "Events",
                    HNLAnalysis_tte(region),
                    processor.iterative_executor,
                    {"schema": NanoAODSchema, 'workers': 6},
                )
                self.output().dump(result)

            if  channel == 'ttm':
                result = processor.run_uproot_job(
                    samples_list,
                    "Events",
                    HNLAnalysis_ttm(region),
                    processor.iterative_executor,
                    {"schema": NanoAODSchema, 'workers': 6},
                )
                self.output().dump(result)
        
            if  channel == 'tem_SS':
                result = processor.run_uproot_job(
                    samples_list,
                    "Events",
                    HNLAnalysis_tem_SS(region),
                    processor.iterative_executor,
                    {"schema": NanoAODSchema, 'workers': 6},
                )
                self.output().dump(result)

            if  channel == 'tem_OS':
                result = processor.run_uproot_job(
                    samples_list,
                    "Events",
                    HNLAnalysis_tem_OS(region),
                    processor.iterative_executor,
                    {"schema": NanoAODSchema, 'workers': 6},
                )
                self.output().dump(result)

        if self.branch == 24:
            Drell_Yann_samples = [
                'DYJetsToLL_M-50',
                'DYJetsToLL_0J',
                'DYJetsToLL_1J',
                'DYJetsToLL_2J',
                'DYJetsToLL_LHEFilterPtZ-0To50',
                'DYJetsToLL_LHEFilterPtZ-50To100',
                'DYJetsToLL_LHEFilterPtZ-100To250',
                'DYJetsToLL_LHEFilterPtZ-250To400',
                'DYJetsToLL_LHEFilterPtZ-400To650',
                'DYJetsToLL_LHEFilterPtZ-650ToInf']

            WJets_samples= [
                'WJetsToLNu',
                'W1JetsToLNu',
                'W2JetsToLNu',
                'W3JetsToLNu',
                'W4JetsToLNu',
                'WJetsToLNu_HT-70To100',
                'WJetsToLNu_HT-100To200',
                'WJetsToLNu_HT-200To400',
                'WJetsToLNu_HT-400To600',
                'WJetsToLNu_HT-600To800',
                'WJetsToLNu_HT-800To1200',
                'WJetsToLNu_HT-1200To2500',
                'WJetsToLNu_HT-2500ToInf']

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
