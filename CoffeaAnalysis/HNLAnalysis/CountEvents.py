from coffea import processor
import awkward as ak
import numpy as np
from CoffeaAnalysis.HNLAnalysis.correction_helpers import get_pileup_correction

class CountEvents(processor.ProcessorABC):
    '''
    Coffea processor that accumulates the sum of weights of Events selected during nanoAOD selection (use "Events" Tree)
    For stitched samples, genWeight are set to +1/-1/0 depending on the sign of genWeight
    '''
    def __init__(self, Backgrounds_stitched, period):
        self._accumulator = processor.dict_accumulator({
            'sumw':processor.defaultdict_accumulator(float),
            'sumw_PUcorr':processor.defaultdict_accumulator(float),
            'sumw_PUcorrUp':processor.defaultdict_accumulator(float),
            'sumw_PUcorrDown':processor.defaultdict_accumulator(float)
        })
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
        output = self.accumulator.identity()
        dataset = events.metadata['dataset']
        if dataset in self.Backgrounds_stitched_list:
            np.asarray(events.genWeight)[events.genWeight < 0] = -1.
            np.asarray(events.genWeight)[events.genWeight > 0] = 1.
            np.asarray(events.genWeight)[events.genWeight == 0] = 0.
            #print(f'Weights of the stitched sample {dataset} are set to -1/0/1')

        output['sumw'][dataset] += ak.sum(events.genWeight)
        corr, corr_up, corr_down = get_pileup_correction(events, self.period, save_updown=False)
        output['sumw_PUcorr'][dataset] += ak.sum(events.genWeight*corr)
        output['sumw_PUcorrUp'][dataset] += ak.sum(events.genWeight*corr_up)
        output['sumw_PUcorrDown'][dataset] += ak.sum(events.genWeight*corr_down)

        return output
    
    def postprocess(self, accumulator):
        return accumulator
