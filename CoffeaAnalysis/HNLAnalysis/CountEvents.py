from coffea import processor
import awkward as ak
import numpy as np
from CoffeaAnalysis.HNLAnalysis.correction_helpers import get_pileup_correction
from collections import defaultdict
import copy

class CountEvents(processor.ProcessorABC):
    '''
    Coffea processor that accumulates the sum of weights of Events selected during nanoAOD selection (use "Events" Tree)
    For stitched samples, genWeight are set to +1/-1/0 depending on the sign of genWeight
    '''
    def __init__(self, Backgrounds_stitched, period):
        self.acc_dict = {
            'sumw':defaultdict(float),
            'sumw_PUcorr':defaultdict(float),
            'sumw_PUcorrUp':defaultdict(float),
            'sumw_PUcorrDown':defaultdict(float)
        }
        self._accumulator = self.acc_dict

        Backgrounds_stitched_list = []
        for key, value in Backgrounds_stitched.items():
            if Backgrounds_stitched[key] is None or len(Backgrounds_stitched[key]) == 0:
                raise 'Missing Backgrounds stitched list'
            Backgrounds_stitched_list += value

        self.Backgrounds_stitched_list = Backgrounds_stitched_list

        self.period = period

    @property
    def accumulator(self):
        return self._accumulator
    
    def process(self, events):
        output = copy.deepcopy(self._accumulator)
        dataset = events.metadata['dataset']
        print(f'Processing {dataset}')
        events_genW = events.genWeight
        if dataset in self.Backgrounds_stitched_list:
            events_genW = events_genW/np.abs(events_genW)
            #print(f'Weights of the stitched sample {dataset} are set to -1/0/1')

        output['sumw'][dataset] += ak.sum(events_genW)
        corr, corr_up, corr_down = get_pileup_correction(events, self.period, save_updown=False)
        output['sumw_PUcorr'][dataset] += ak.sum(events_genW*corr)

        return output
    
    def postprocess(self, accumulator):
        return accumulator
