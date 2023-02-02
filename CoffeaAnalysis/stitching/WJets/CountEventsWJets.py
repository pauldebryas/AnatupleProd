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
            'sumw0Jets':processor.defaultdict_accumulator(float),# sumw of events with LHE.Njets == 0
            'sumw1Jets':processor.defaultdict_accumulator(float),# sumw of events with LHE.Njets == 1
            'sumw2Jets':processor.defaultdict_accumulator(float),# sumw of events with LHE.Njets == 2
            'sumw3Jets':processor.defaultdict_accumulator(float),# sumw of events with LHE.Njets == 3
            'sumw4Jets':processor.defaultdict_accumulator(float) # sumw of events with LHE.Njets == 4
        })

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
        output['sumw0Jets'][dataset] += ak.sum(events.genWeight[events.LHE.Njets == 0])
        output['sumw1Jets'][dataset] += ak.sum(events.genWeight[events.LHE.Njets == 1])
        output['sumw2Jets'][dataset] += ak.sum(events.genWeight[events.LHE.Njets == 2])
        output['sumw3Jets'][dataset] += ak.sum(events.genWeight[events.LHE.Njets == 3])
        output['sumw4Jets'][dataset] += ak.sum(events.genWeight[events.LHE.Njets == 4])
        return output
    
    def postprocess(self, accumulator):
        return accumulator

class CountEventsHT(processor.ProcessorABC):
    '''
    Coffea processor that accumulates the original sum of weights
    of events for different HT bins at LHE level. 
    Works for MC events only.
    '''
    def __init__(self):
        self._accumulator = processor.dict_accumulator({
            'sumw':processor.defaultdict_accumulator(float),
            'sumw0to70':processor.defaultdict_accumulator(float),
            'sumw70to100':processor.defaultdict_accumulator(float),
            'sumw100to200':processor.defaultdict_accumulator(float),
            'sumw200to400':processor.defaultdict_accumulator(float),
            'sumw400to600':processor.defaultdict_accumulator(float),
            'sumw600to800':processor.defaultdict_accumulator(float),
            'sumw800to1200':processor.defaultdict_accumulator(float),
            'sumw1200to2500':processor.defaultdict_accumulator(float),
            'sumw2500toInf':processor.defaultdict_accumulator(float)
        })

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
        output['sumw0to70'][dataset] += ak.sum(events.genWeight[(events.LHE.HT >= 0) & (events.LHE.HT < 70)])
        output['sumw70to100'][dataset] += ak.sum(events.genWeight[(events.LHE.HT >= 70) & (events.LHE.HT < 100)])
        output['sumw100to200'][dataset] += ak.sum(events.genWeight[(events.LHE.HT >= 100) & (events.LHE.HT < 200)])
        output['sumw200to400'][dataset] += ak.sum(events.genWeight[(events.LHE.HT >= 200) & (events.LHE.HT < 400)])
        output['sumw400to600'][dataset] += ak.sum(events.genWeight[(events.LHE.HT >= 400) & (events.LHE.HT < 600)])
        output['sumw600to800'][dataset] += ak.sum(events.genWeight[(events.LHE.HT >= 600) & (events.LHE.HT < 800)])
        output['sumw800to1200'][dataset] += ak.sum(events.genWeight[(events.LHE.HT >= 800) & (events.LHE.HT < 1200)])
        output['sumw1200to2500'][dataset] += ak.sum(events.genWeight[(events.LHE.HT >= 1200) & (events.LHE.HT < 2500)])
        output['sumw2500toInf'][dataset] += ak.sum(events.genWeight[(events.LHE.HT >= 2500)])
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

class CountEventsNJetsHT(processor.ProcessorABC):
    '''
    Coffea processor that accumulates the original sum of weights
    of events for different HT/NJets bins at LHE level. 
    Works for MC events only.
    '''
    def __init__(self):
        dict = {}
        dict['sumw'] = processor.defaultdict_accumulator(float)
        for NJets in self.get_NJets_bins():
            for HTbin in self.get_HT_bins():
                dict[f'sumw_HT-{HTbin}_Jets-{NJets}'] = processor.defaultdict_accumulator(float)

        self._accumulator = processor.dict_accumulator(dict)

    @staticmethod
    def get_NJets_bins():
        return [
            '0',
            '1',
            '2',
            '3',
            '4'
        ]

    @staticmethod
    def get_HT_bins():
        return [
            '0',
            '0to70',
            '70to100',
            '100to200',
            '200to400',
            '400to600',
            '600to800',
            '800to1200',
            '1200to2500',
            '2500toInf'
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
            output[f'sumw_HT-0_Jets-{NJets}'][dataset] += ak.sum(events.genWeight[(events.LHE.HT == 0) & (events.LHE.Njets == int(NJets))])
            output[f'sumw_HT-0to70_Jets-{NJets}'][dataset] += ak.sum(events.genWeight[(events.LHE.HT > 0) & (events.LHE.HT < 70) & (events.LHE.Njets == int(NJets))])
            output[f'sumw_HT-70to100_Jets-{NJets}'][dataset] += ak.sum(events.genWeight[(events.LHE.HT >= 70) & (events.LHE.HT < 100) & (events.LHE.Njets == int(NJets))])
            output[f'sumw_HT-100to200_Jets-{NJets}'][dataset] += ak.sum(events.genWeight[(events.LHE.HT >= 100) & (events.LHE.HT < 200) & (events.LHE.Njets == int(NJets))])
            output[f'sumw_HT-200to400_Jets-{NJets}'][dataset] += ak.sum(events.genWeight[(events.LHE.HT >= 200) & (events.LHE.HT < 400) & (events.LHE.Njets == int(NJets))])
            output[f'sumw_HT-400to600_Jets-{NJets}'][dataset] += ak.sum(events.genWeight[(events.LHE.HT >= 400) & (events.LHE.HT < 600) & (events.LHE.Njets == int(NJets))])
            output[f'sumw_HT-600to800_Jets-{NJets}'][dataset] += ak.sum(events.genWeight[(events.LHE.HT >= 600) & (events.LHE.HT < 800) & (events.LHE.Njets == int(NJets))])
            output[f'sumw_HT-800to1200_Jets-{NJets}'][dataset] += ak.sum(events.genWeight[(events.LHE.HT >= 800) & (events.LHE.HT < 1200) & (events.LHE.Njets == int(NJets))])
            output[f'sumw_HT-1200to2500_Jets-{NJets}'][dataset] += ak.sum(events.genWeight[(events.LHE.HT >= 1200) & (events.LHE.HT < 2500) & (events.LHE.Njets == int(NJets))])
            output[f'sumw_HT-2500toInf_Jets-{NJets}'][dataset] += ak.sum(events.genWeight[(events.LHE.HT >= 2500) & (events.LHE.Njets == int(NJets))])

        return output
    
    def postprocess(self, accumulator):
        return accumulator