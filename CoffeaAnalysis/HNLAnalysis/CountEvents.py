from coffea import processor
import awkward as ak
import numpy as np
from CoffeaAnalysis.HNLAnalysis.correction_helpers import get_pileup_correction

class CountEvents(processor.ProcessorABC):
    '''
    Coffea processor that accumulates the sum of weights of Events selected during nanoAOD selection (use "Events" Tree)
    For stitched samples, genWeight are set to +1/-1/0 depending on the sign of genWeight
    '''
    def __init__(self, Backgrounds_stitched):
        self._accumulator = processor.dict_accumulator({
            'sumw':processor.defaultdict_accumulator(float),
            'sumw_PUcorr':processor.defaultdict_accumulator(float)
        })

        if Backgrounds_stitched is None or len(Backgrounds_stitched) == 0:
            raise 'Missing Backgrounds stitched list'
        self.Backgrounds_stitched = Backgrounds_stitched

    @property
    def accumulator(self):
        return self._accumulator
    
    def process(self, events):
        output = self.accumulator.identity()
        dataset = events.metadata['dataset']
        if '2018' not in dataset:
            if dataset in self.Backgrounds_stitched:
                np.asarray(events.genWeight)[events.genWeight < 0] = -1.
                np.asarray(events.genWeight)[events.genWeight > 0] = 1.
                np.asarray(events.genWeight)[events.genWeight == 0] = 0.
                print(dataset)
                print('stitched sample')

            output['sumw'][dataset] += ak.sum(events.genWeight)
            corr = get_pileup_correction(events, 'nominal')
            output['sumw_PUcorr'][dataset] += ak.sum(events.genWeight*corr)
        else:
            #For data samples, we need to find the original number of event in the unskimed file --> to be modified
            dataset = dataset[0:-1]
            output['sumw'][dataset] += len(events)

        return output
    
    def postprocess(self, accumulator):
        return accumulator
