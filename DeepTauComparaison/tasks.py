#!/usr/bin/env python
import law
import luigi
import pickle
import os

from coffea import processor
from coffea.nanoevents import NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = True

from DeepTauComparaison.HNLAnalysis_ttm_new import HNLAnalysis_ttm_new
from DeepTauComparaison.HNLAnalysis_ttm_old import HNLAnalysis_ttm_old
from CoffeaAnalysis.HNLAnalysis.helpers import files_from_path

from run_tools.law_customizations import Task, HTCondorWorkflow

class RunComparison(Task, HTCondorWorkflow, law.LocalWorkflow):
    # tag that will be added to produced pkl files and not mixt up with different analysis
    tag = luigi.Parameter()

    def create_branch_map(self):
        path = os.path.join(self.ana_path(), 'DeepTauComparaison', 'results', self.tag)
        output_files = [
            path + '/result_ttm_new_regionA.pkl',
            path + '/result_ttm_old_regionA.pkl',
            path + '/result_ttm_new_regionB.pkl',
            path + '/result_ttm_old_regionB.pkl',
            path + '/result_ttm_new_regionC.pkl',
            path + '/result_ttm_old_regionC.pkl',
            path + '/result_ttm_new_regionD.pkl',
            path + '/result_ttm_old_regionD.pkl']
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

        if self.branch in [0,1]:
            region = 'A'
        if self.branch in [2,3]:
            region = 'B'
        if self.branch in [4,5]:
            region = 'C'
        if self.branch in [6,7]:
            region = 'D'

        if self.branch in [0,2,4,6]:
            channel = 'ttm_new'
        if self.branch in [1,3,5,7]:
            channel = 'ttm_old'

        if region not in ['A','B','C','D']:
            raise RuntimeError("Incorrect region name detected: {}".format(region))

        if channel not in ['ttm_new','ttm_old']:
            raise RuntimeError("Incorrect channel name detected: {}".format(channel))

        if  channel == 'ttm_new':
            result = processor.run_uproot_job(
                samples_list,
                "Events",
                HNLAnalysis_ttm_new(region),
                processor.iterative_executor,
                {"schema": NanoAODSchema, 'workers': 6},
            )
            self.output().dump(result)

        if  channel == 'ttm_old':
            result = processor.run_uproot_job(
                samples_list,
                "Events",
                HNLAnalysis_ttm_old(region),
                processor.iterative_executor,
                {"schema": NanoAODSchema, 'workers': 6},
            )
            self.output().dump(result)
