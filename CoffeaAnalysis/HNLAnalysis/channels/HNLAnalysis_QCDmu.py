import numpy as np
import awkward as ak
import os
from coffea import processor
from collections import defaultdict
import copy

from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_lepton, save_Event
from CoffeaAnalysis.HNLAnalysis.helpers import delta_r
from CoffeaAnalysis.HNLAnalysis.HNLProcessor import HNLProcessor

class HNLAnalysis_QCDmu(processor.ProcessorABC, HNLProcessor):
    def __init__(self, stitched_list, tag, xsecs, periods, dataHLT, debugMode):
        HNLProcessor.__init__(self, stitched_list, tag, xsecs, periods, dataHLT, debugMode)
        self.acc_dict = {}
        self.selections = self.get_selections()
        for selection in self.selections:
            self.acc_dict[f'n_ev_{selection}'] = defaultdict(int)
            self.acc_dict[f'sumw_{selection}'] = defaultdict(int)
        self._accumulator = self.acc_dict

    @property
    def accumulator(self):
        return self._accumulator

    @staticmethod
    def get_selections():
        return [
            'init',
            'reweight',
            'MET_Filter',
            'Exactly1mu',
            'HLT',
            'DRJet'
        ]

    # we will receive a NanoEvents
    def process(self, events):

        out = copy.deepcopy(self._accumulator)
        events, out = self.init_process(out, events)

        # Do the general lepton selection
        events_QCDmu = self.Lepton_selection(events)

        # Apply the cuts and select leptons
        events_QCDmu = self.analyse_QCDmu(events_QCDmu, out)

        # Save anatuple
        save_file, lst = self.save_anatuple_QCDmu(events_QCDmu, self.tag)
        
        save_Event(save_file, lst, 'Events')

        return out

    def analyse_QCDmu(self, events, out= None):

        # select events: require at least 1 reco mu 
        events_QCDmu = events[(ak.num(events.SelElectron) == 0) & (ak.num(events.SelMuon) == 1)]

        if out != None:
            out[f'sumw_Exactly1mu'][self.ds] += ak.sum(events_QCDmu.genWeight)
            out[f'n_ev_Exactly1mu'][self.ds] += len(events_QCDmu)

        # events should pass HLT requirements
        if self.period == '2018':
            events_QCDmu = events_QCDmu[events_QCDmu.HLT.Mu20 | events_QCDmu.HLT.Mu27 | events_QCDmu.HLT.Mu50 | events_QCDmu.HLT.Mu55]
        if self.period == '2017':
            events_QCDmu = events_QCDmu[events_QCDmu.HLT.Mu20 | events_QCDmu.HLT.Mu27 | events_QCDmu.HLT.Mu50 | events_QCDmu.HLT.Mu55]
        if self.period == '2016':
            events_QCDmu = events_QCDmu[events_QCDmu.HLT.Mu20 | events_QCDmu.HLT.Mu27 | events_QCDmu.HLT.Mu50 | events_QCDmu.HLT.Mu55]
        if self.period == '2016_HIPM':
            events_QCDmu = events_QCDmu[events_QCDmu.HLT.Mu20 | events_QCDmu.HLT.Mu27 | events_QCDmu.HLT.Mu50 | events_QCDmu.HLT.Mu55]

        if out != None:
            out[f'sumw_HLT'][self.ds] += ak.sum(events_QCDmu.genWeight)
            out[f'n_ev_HLT'][self.ds] += len(events_QCDmu)

        # Save bjets candidates
        jets_candidates = events_QCDmu.Jet[(events_QCDmu.Jet.pt > 30.) & (events_QCDmu.Jet.eta < 2.5) & (events_QCDmu.Jet.eta > -2.5)]
        Sel_Muon = events_QCDmu.SelMuon[:,0]
        drcut_jetslep = delta_r(Sel_Muon,jets_candidates) > 0.7
        jets_candidates = jets_candidates[drcut_jetslep]
        
        events_QCDmu['nJets'] = ak.num(jets_candidates)
        events_QCDmu = events_QCDmu[ak.num(jets_candidates) >= 1]

        if out != None:
            out[f'sumw_DRJet'][self.ds] += ak.sum(events_QCDmu.genWeight)
            out[f'n_ev_DRJet'][self.ds] += len(events_QCDmu)

        return events_QCDmu

    
    def save_anatuple_QCDmu(self, events, tag):

        exclude_list = ['genPartIdx']
    
        path = f'/afs/cern.ch/work/p/pdebryas/HNL/tmp/{self.period}/{tag}/QCDmu/{self.ds}/'

        if not os.path.exists(path):
            os.makedirs(path)

        save_file = path + f'{self.ds}_anatuple_0.root'

        i = 0
        while os.path.isfile(save_file):
            i = i+1
            save_file = path + f'{self.ds}_anatuple_{str(i)}.root'

        lst = { "event": np.array(events.event),
                "genWeight": np.array(events.genWeight),
                "luminosityBlock": np.array(events.luminosityBlock),
                "run": np.array(events.run),
                "MET_pt": np.array(events.MET['pt']),
                "MET_phi": np.array(events.MET['phi']),
                "nJets": np.array(events.nJets)
            }

        if self.mode == 'signal':
            lst['HNLmass'] = np.ones(len(events))*int(self.ds[self.ds.rfind("-") + 1:])

        lst = save_anatuple_lepton(events.SelMuon, lst, exclude_list, 'Muon')
 
        return save_file, lst

    def postprocess(self, accumulator):
        return accumulator
    
