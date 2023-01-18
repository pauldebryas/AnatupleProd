from coffea import processor
import awkward as ak
import numpy as np

class CountEvents(processor.ProcessorABC):
    '''Coffea processor that accumulates the original sum of weights
    for different NanoAOD samples with pre-selections, which are stored 
    in the  "Runs" tree. 

    Works for MC events (backgrounds and signal), but for data take the len of the skimed
    files so not good: so please not use it for data.
    '''
    def __init__(self):
        self._accumulator = processor.dict_accumulator({
            'sumw':processor.defaultdict_accumulator(float)
        })

    @property
    def accumulator(self):
        return self._accumulator
    
    def process(self, events):
        output = self.accumulator.identity()
        dataset = events.metadata['dataset']
        if '2018' not in dataset:
            output['sumw'][dataset] += ak.sum(events.genEventSumw)
        else:
            #For data samples, we need to find the original number of event in the unskimed file --> to be modified
            dataset = dataset[0:-1]
            output['sumw'][dataset] += len(events)

        return output
    
    def postprocess(self, accumulator):
        return accumulator

class CountEvents_stitched_samples(processor.ProcessorABC):
    '''
    Coffea processor that accumulates the original sum of weights fixed for stitched samples
    +1/-1/0 depending on the sign of genWeight
    '''
    def __init__(self):
        self._accumulator = processor.dict_accumulator({
            'sumw_fixed':processor.defaultdict_accumulator(float)
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        dataset = events.metadata['dataset']

        np.asarray(events.genWeight)[events.genWeight < 0] = -1.
        np.asarray(events.genWeight)[events.genWeight > 0] = 1.
        np.asarray(events.genWeight)[events.genWeight == 0] = 0.

        output['sumw_fixed'][dataset] += ak.sum(events.genWeight)

        return output
    
    def postprocess(self, accumulator):
        return accumulator