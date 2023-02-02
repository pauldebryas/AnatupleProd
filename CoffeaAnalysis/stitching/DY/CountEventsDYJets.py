from coffea import processor
import awkward as ak
import numpy as np

class CountEventsNJets(processor.ProcessorABC):
    '''
    Coffea processor that accumulates the original sum of weights
    of events for different number of jets at LHE. 
    Works for MC events only.
    '''
    def __init__(self):
        self._accumulator = processor.dict_accumulator({
            'sumw':processor.defaultdict_accumulator(float),     # total sumw
            'sumw0Jets':processor.defaultdict_accumulator(float),# sumw of events with LHE.NpNLO == 0
            'sumw1Jets':processor.defaultdict_accumulator(float),# sumw of events with LHE.NpNLO == 1
            'sumw2Jets':processor.defaultdict_accumulator(float) # sumw of events with LHE.NpNLO == 2
        })

    @property
    def accumulator(self):
        return self._accumulator
    
    def process(self, events):
        output = self.accumulator.identity()
        dataset = events.metadata['dataset']

        # an event should have weights = +1/0/-1 (weights not coherent in between samples)
        sumw_w = np.ones(len(events))
        sumw_w[events.genWeight < 0] = -1.
        sumw_w[events.genWeight == 0] = 0.
        events.genWeight = ak.Array(sumw_w)
        
        output['sumw'][dataset] += ak.sum(events.genWeight)
        output['sumw0Jets'][dataset] += ak.sum(events.genWeight[events.LHE.NpNLO == 0])
        output['sumw1Jets'][dataset] += ak.sum(events.genWeight[events.LHE.NpNLO == 1])
        output['sumw2Jets'][dataset] += ak.sum(events.genWeight[events.LHE.NpNLO == 2])
        return output
    
    def postprocess(self, accumulator):
        return accumulator
    
class CountEvents(processor.ProcessorABC):
    '''
    Coffea processor that accumulates the original sum of weights
    for different NanoAOD samples with pre-selections, which are stored 
    in the  "Runs" tree. 

    Works for MC events only.
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
        output['sumw'][dataset] += ak.sum(events.genEventSumw)
        return output
    
    def postprocess(self, accumulator):
        return accumulator

class CountEventsNJetsPtZ(processor.ProcessorABC):
    '''
    Coffea processor that accumulates the original sum of weights
    of events for different PtZ/NJets bins at LHE level. 
    Works for MC events only.
    '''
    def __init__(self):
        dict = {}
        dict['sumw'] = processor.defaultdict_accumulator(float)
        for NJets in self.get_NJets_bins():
            for PtZbin in self.get_PtZ_bins():
                dict[f'sumw_PtZ-{PtZbin}_Jets-{NJets}'] = processor.defaultdict_accumulator(float)

        self._accumulator = processor.dict_accumulator(dict)

    @staticmethod
    def get_NJets_bins():
        return [
            '0',
            '1',
            '2'
        ]

    @staticmethod
    def get_PtZ_bins():
        return [
            '0',
            '0to50',
            '50to100',
            '100to250',
            '250to400',
            '400to650',
            '650toInf'
        ]

    @property
    def accumulator(self):
        return self._accumulator
    
    def process(self, events):
        output = self.accumulator.identity()
        dataset = events.metadata['dataset']
        
        # an event should have weights = +1/0/-1 (weights not coherent in between samples)
        np.asarray(events.genWeight)[events.genWeight < 0] = -1.
        np.asarray(events.genWeight)[events.genWeight > 0] = 1.
        np.asarray(events.genWeight)[events.genWeight == 0] = 0.

        output['sumw'][dataset] += ak.sum(events.genWeight)

        for NJets in self.get_NJets_bins():
            output[f'sumw_PtZ-0_Jets-{NJets}'][dataset] += ak.sum(events.genWeight[(events.LHE.Vpt == 0) & (events.LHE.NpNLO == int(NJets))])
            output[f'sumw_PtZ-0to50_Jets-{NJets}'][dataset] += ak.sum(events.genWeight[(events.LHE.Vpt > 0) & (events.LHE.Vpt < 50) & (events.LHE.NpNLO == int(NJets))])
            output[f'sumw_PtZ-50to100_Jets-{NJets}'][dataset] += ak.sum(events.genWeight[(events.LHE.Vpt >= 50) & (events.LHE.Vpt < 100) & (events.LHE.NpNLO == int(NJets))])
            output[f'sumw_PtZ-100to250_Jets-{NJets}'][dataset] += ak.sum(events.genWeight[(events.LHE.Vpt >= 100) & (events.LHE.Vpt < 250) & (events.LHE.NpNLO == int(NJets))])
            output[f'sumw_PtZ-250to400_Jets-{NJets}'][dataset] += ak.sum(events.genWeight[(events.LHE.Vpt >= 250) & (events.LHE.Vpt < 400) & (events.LHE.NpNLO == int(NJets))])
            output[f'sumw_PtZ-400to650_Jets-{NJets}'][dataset] += ak.sum(events.genWeight[(events.LHE.Vpt >= 400) & (events.LHE.Vpt < 650) & (events.LHE.NpNLO == int(NJets))])
            output[f'sumw_PtZ-650toInf_Jets-{NJets}'][dataset] += ak.sum(events.genWeight[(events.LHE.Vpt >= 650) & (events.LHE.NpNLO == int(NJets))])

        return output
    
    def postprocess(self, accumulator):
        return accumulator