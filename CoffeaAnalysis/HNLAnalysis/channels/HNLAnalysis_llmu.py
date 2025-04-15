import numpy as np
import awkward as ak
from coffea import processor
from collections import defaultdict
import copy
import os

from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_lepton, save_Event, add_gen_matching_info, bjet_candidates
from CoffeaAnalysis.HNLAnalysis.helpers import OSOF_sel, FinalFakeCandidate_sel
from CoffeaAnalysis.HNLAnalysis.HNLProcessor import HNLProcessor

class HNLAnalysis_llmu(processor.ProcessorABC, HNLProcessor):
    def __init__(self, stitched_list, tag, xsecs, periods, dataHLT, debugMode, sample_name):
        HNLProcessor.__init__(self, stitched_list, tag, xsecs, periods, dataHLT, debugMode)
        self.acc_dict = {}
        self.selections = self.get_selections()
        for selection in self.selections:
            self.acc_dict[f'n_ev_{selection}'] = defaultdict(int)
            self.acc_dict[f'sumw_{selection}'] = defaultdict(int)
        self._accumulator = self.acc_dict
        self.sample_name = sample_name

    @property
    def accumulator(self):
        return self._accumulator

    @staticmethod
    def get_selections():
        return [
            'init',
            'reweight',
            'MET_Filter',
            'lllsel',
            'emusel',
            'muSel'
        ]

    # we will receive a NanoEvents
    def process(self, events):

        out = copy.deepcopy(self._accumulator)
        events, out = self.init_process(out, events)

        # Do the general lepton selection
        events_llmu = self.Lepton_selection(events)

        # Apply the cuts and select leptons
        events_llmu, lepton1, lepton2, Sel_Mu = self.analyse_llmu(events_llmu, out)

        # save GenPart info in case MC sample
        if self.mode != 'Data':
            lepton1 = add_gen_matching_info(events_llmu, lepton1)
            lepton2 = add_gen_matching_info(events_llmu, lepton2)
            Sel_Mu  = add_gen_matching_info(events_llmu, Sel_Mu)

        # Save anatuple
        print(f'Saving {len(events_llmu)} events')
        save_file, lst = self.save_anatuple_llmu(events_llmu, lepton1, lepton2, Sel_Mu, self.tag)
        
        save_Event(save_file, lst, 'Events')

        return out

    def analyse_llmu(self, events, out):
        # l1 and l2 should be OSOF light lepton and m(l1,l2) away from m_Z
        # l3 is a mu 

        # select lll events: require at least 3 reco mu or 2 reco e and 1 reco mu
        events_llmu = events[(ak.num(events.SelElectron) >= 1) & (ak.num(events.SelMuon) >= 2)]
        out[f'sumw_lllsel'][self.ds] += ak.sum(events_llmu.genWeight)
        out[f'n_ev_lllsel'][self.ds] += len(events_llmu)

        '''
        Filters events with two light leptons satisfying:
            - Opposite charges
            - Opposite flavor
            - l1/l2 iso < 0.15 
        Selects the pair with smalest mean isolation.
        '''
        events_llmu, lepton1, lepton2 = OSOF_sel(events_llmu)

        out[f'sumw_emusel'][self.ds] += ak.sum(events_llmu.genWeight)
        out[f'n_ev_emusel'][self.ds] += len(events_llmu)

        # select Mu candidate with dr(l1,Muon)>0.5 and dr(l2,Muon)>0.5, and take the mu with highest pt
        events_llmu, lepton1, lepton2, Sel_Mu = FinalFakeCandidate_sel(events_llmu, lepton1, lepton2, 'muon')

        out[f'sumw_muSel'][self.ds] += ak.sum(events_llmu.genWeight)
        out[f'n_ev_muSel'][self.ds] += len(events_llmu)

        # Save bjets candidates
        bjet_candidates(events_llmu, lepton1, lepton2, Sel_Mu, self.period)

        return events_llmu, lepton1, lepton2, Sel_Mu
    
    def save_anatuple_llmu(self, events, lepton1, lepton2, SelMuon, tag):

        path = f'/afs/cern.ch/work/p/pdebryas/HNL/tmp/{self.period}/{tag}/llmu/{self.sample_name}/'

        if not os.path.exists(path):
            os.makedirs(path)

        save_file = path + f'{self.sample_name}_anatuple_0.root'

        i = 0
        while os.path.isfile(save_file):
            i = i+1
            save_file = path + f'{self.sample_name}_anatuple_{str(i)}.root'


        lst = { "event": np.array(events.event),
                "genWeight": np.array(events.genWeight),
                "luminosityBlock": np.array(events.luminosityBlock),
                "run": np.array(events.run),
                "MET_pt": np.array(events.SelMET['pt']),
                "MET_phi": np.array(events.SelMET['phi']),
                "nbjetsLoose": np.array(events.nbjetsLoose),
            }

        if self.mode == 'signal':
            lst['HNLmass'] = np.ones(len(events))*int(self.ds[self.ds.rfind("-") + 1:])

        exclude_list =  ([f for f in events.Electron.fields if f not in events.Muon.fields] + [f for f in events.Muon.fields if f not in events.Electron.fields])

        lst = save_anatuple_lepton(lepton1, lst, exclude_list, 'Lepton1')

        lst = save_anatuple_lepton(lepton2, lst, exclude_list, 'Lepton2')

        lst = save_anatuple_lepton(SelMuon, lst, ['genPartIdx'], 'Muon')

        return save_file, lst

    def postprocess(self, accumulator):
        return accumulator

