#!/usr/bin/env python
import law
import luigi
import pickle
import os

from coffea import processor
from coffea.nanoevents import NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = True

from CoffeaAnalysis.CountEvents import CountEvents
from CoffeaAnalysis.HNLAnalysis import HNLAnalysis
from CoffeaAnalysis.helpers import files_from_path
#from CoffeaAnalysis.samples import signal_samples, Data_samples, MCbackground_samples
from NanoProd.tasks import CreateNanoSkims

from run_tools.law_customizations import Task, HTCondorWorkflow

class RunAnalysis(Task, HTCondorWorkflow, law.LocalWorkflow):
    # tag that will be added to produced pkl files and not mixt up with different analysis
    tag = luigi.Parameter(default='TEST2')
    # requires skimmed NanoAOD files produce with CreateNanoSkims task
    def workflow_requires(self):
        return {"Nano_Skim": CreateNanoSkims.req(self) }
    
    def requires(self):
        return self.workflow_requires()

    def create_branch_map(self):
        path = os.path.join(self.ana_path(), 'CoffeaAnalysis', 'results')
        output_files = [path + '/counter_{}.pkl'.format(self.tag), path + '/result_{}.pkl'.format(self.tag)]
        branches = {}
        branches[0] = output_files[0]
        branches[1] = output_files[1]
        return branches
    
    def output(self):
        output_path = self.branch_data
        # output two pkl files
        return self.local_analysis_target(output_path)

    def run(self):
        output_path = self.branch_data
        tag = self.tag

        self.load_sample_configs()
        samples_list = {}
        for period, samples in self.samples.items():
            for sample_name in sorted(samples.keys()):
                sampleType = samples[sample_name].get('sampleType', None)
                if sampleType is None or len(sampleType) == 0:
                    self.publish_message("Missing sampleType for sample: {}".format(sample_name))
                    raise RuntimeError("Missing sampleType has been detected.")

                path_to_sample = os.path.join(self.central_path(), self.version, 'nanoAOD', period, sampleType, sample_name + '.root')
                if os.path.exists(path_to_sample):
                    samples_list[sample_name] = files_from_path(os.path.join(self.central_path(), self.version, 'nanoAOD', period, sampleType, sample_name + '.root'))
                else:
                    samples_list[sample_name] = files_from_path(os.path.join(self.central_path(), self.version, 'nanoAOD', period, sampleType, sample_name))

        if  self.branch == 0:
            self.publish_message('processing ' + 'counter_{}.pkl'.format(self.tag) + ' file:')

            event_counter = processor.run_uproot_job(
                samples_list,
                'Runs',
                CountEvents(),
                #processor.futures_executor,
                processor.iterative_executor, # may be better for debugging

                {"schema": NanoAODSchema, 'workers': 6},
            )

            self.output().dump(event_counter)

        if  self.branch == 1:
            self.publish_message('processing ' + 'result_{}.pkl'.format(self.tag) + ' file:')

            result = processor.run_uproot_job(
                samples_list,
                "Events",
                HNLAnalysis(),
                #processor.futures_executor,
                processor.iterative_executor, # may be better for debugging
                {"schema": NanoAODSchema, 'workers': 6},
            )

            self.output().dump(result)
