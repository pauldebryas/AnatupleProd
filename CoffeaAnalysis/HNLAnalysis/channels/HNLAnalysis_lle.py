import numpy as np
import awkward as ak
from coffea import processor
from collections import defaultdict
import copy
import os

from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_lepton, save_Event, add_gen_matching_info, bjet_candidates
from CoffeaAnalysis.HNLAnalysis.helpers import OSOF_sel, FinalFakeCandidate_sel
from CoffeaAnalysis.HNLAnalysis.HNLProcessor import HNLProcessor

class HNLAnalysis_lle(processor.ProcessorABC, HNLProcessor):
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
            'eSel'
        ]

    # we will receive a NanoEvents
    def process(self, events):

        out = copy.deepcopy(self._accumulator)
        events, out = self.init_process(out, events)

        # Do the general lepton selection
        events_lle = self.Lepton_selection(events)

        # Apply the cuts and select leptons
        events_lle, lepton1, lepton2, Sel_e = self.analyse_lle(events_lle, out)

        # save GenPart info in case MC sample
        if self.mode != 'Data':
            lepton1 = add_gen_matching_info(events_lle, lepton1)
            lepton2 = add_gen_matching_info(events_lle, lepton2)
            Sel_e  = add_gen_matching_info(events_lle, Sel_e)

        # Save anatuple
        print(f'Saving {len(events_lle)} events')
        save_file, lst = self.save_anatuple_lle(events_lle, lepton1, lepton2, Sel_e, self.tag)
        
        save_Event(save_file, lst, 'Events')

        return out

    def analyse_lle(self, events, out):
        # l1 and l2 should be OSOF light lepton and m(l1,l2) away from m_Z
        # l3 is a e

        # select lll events: require at least 3 reco e or 2 reco mu and 1 reco e
        events_lle = events[(ak.num(events.SelMuon) >= 1) & (ak.num(events.SelElectron) >= 2)]
        out[f'sumw_lllsel'][self.ds] += ak.sum(events_lle.genWeight)
        out[f'n_ev_lllsel'][self.ds] += len(events_lle)

        '''
        Filters events with two light leptons satisfying:
            - Opposite charges
            - Opposite flavor
            - l1/l2 iso < 0.15 
        Selects the pair with smalest mean isolation.
        '''
        events_lle, lepton1, lepton2 = OSOF_sel(events_lle)

        out[f'sumw_emusel'][self.ds] += ak.sum(events_lle.genWeight)
        out[f'n_ev_emusel'][self.ds] += len(events_lle)

        # select electron with dr(l1,e)>0.5 and dr(l2,e)>0.5, and take the e with highest pt
        events_lle, lepton1, lepton2, Sel_e = FinalFakeCandidate_sel(events_lle, lepton1, lepton2, 'electron')

        out[f'sumw_eSel'][self.ds] += ak.sum(events_lle.genWeight)
        out[f'n_ev_eSel'][self.ds] += len(events_lle)

        # Save bjets candidates
        bjet_candidates(events_lle, lepton1, lepton2, Sel_e, self.period)

        return events_lle, lepton1, lepton2, Sel_e
    
    def save_anatuple_lle(self, events, lepton1, lepton2, SelElectron, tag):

        path = f'/afs/cern.ch/work/p/pdebryas/HNL/tmp/{self.period}/{tag}/lle/{self.sample_name}/'

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

        lst = save_anatuple_lepton(SelElectron, lst, ['genPartIdx'], 'Electron')

        return save_file, lst

    def postprocess(self, accumulator):
        return accumulator


