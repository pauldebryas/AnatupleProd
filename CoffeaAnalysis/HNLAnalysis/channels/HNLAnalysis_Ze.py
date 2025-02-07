import numpy as np
import awkward as ak
from coffea import processor
from collections import defaultdict
import copy
import os

from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_lepton, save_Event, add_gen_matching_info
from CoffeaAnalysis.HNLAnalysis.helpers import ll_from_Z_sel, FinalLL_sel
from CoffeaAnalysis.HNLAnalysis.HNLProcessor import HNLProcessor

class HNLAnalysis_Ze(processor.ProcessorABC, HNLProcessor):
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
            'lllsel',
            'llinZ',
            'eSel'
        ]

    # we will receive a NanoEvents
    def process(self, events):

        out = copy.deepcopy(self._accumulator)
        events, out = self.init_process(out, events)

        # Do the general lepton selection
        events_Ze = self.Lepton_selection(events)

        # Apply the cuts and select leptons
        events_Ze, lepton1, lepton2, Sel_e = self.analyse_Ze(events_Ze, out)

        # save GenPart info in case MC sample
        if self.mode != 'Data':
            lepton1 = add_gen_matching_info(events_Ze, lepton1)
            lepton2 = add_gen_matching_info(events_Ze, lepton2)
            Sel_e  = add_gen_matching_info(events_Ze, Sel_e)

        # Save anatuple
        print(f'Saving {len(events_Ze)} events')
        save_file, lst = self.save_anatuple_Ze(events_Ze, lepton1, lepton2, Sel_e, self.tag)
        
        save_Event(save_file, lst, 'Events')

        return out

    def analyse_Ze(self, events, out):
        # l1 and l2 should be 2 mu or 2 e with OS and m(l1,l2) ~ m_Z
        # l3 is a e

        # select lll events: require at least 3 reco e or 2 reco mu and 1 reco e
        events_Ze = events[((ak.num(events.SelMuon) >= 2) & (ak.num(events.SelElectron) >= 1)) | (ak.num(events.SelElectron) >= 3)]
        out[f'sumw_lllsel'][self.ds] += ak.sum(events_Ze.genWeight)
        out[f'n_ev_lllsel'][self.ds] += len(events_Ze)

        '''
        Filters events with two leptons (muons or electrons) satisfying:
            - Invariant mass of 91.2 ± 15 GeV
            - Opposite charges
        Selects the pair with invariant mass closest to the Z mass (91.2 GeV).
        '''
        events_Ze, lepton1, lepton2 = ll_from_Z_sel(events_Ze)

        out[f'sumw_llinZ'][self.ds] += ak.sum(events_Ze.genWeight)
        out[f'n_ev_llinZ'][self.ds] += len(events_Ze)

        # select electron with dr(l1,e)>0.5 and dr(l2,e)>0.5 and minimum pfRelIso03_all in case of ambiguity
        events_Ze, lepton1, lepton2, Sel_e = FinalLL_sel(events_Ze, lepton1, lepton2, 'electron')

        out[f'sumw_eSel'][self.ds] += ak.sum(events_Ze.genWeight)
        out[f'n_ev_eSel'][self.ds] += len(events_Ze)

        return events_Ze, lepton1, lepton2, Sel_e
    
    def save_anatuple_Ze(self, events, lepton1, lepton2, SelElectron, tag):

        path = f'/afs/cern.ch/work/p/pdebryas/HNL/tmp/{self.period}/{tag}/Ze/{self.ds}/'

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
                "MET_pt": np.array(events.SelMET['pt']),
                "MET_phi": np.array(events.SelMET['phi']),
                "IsLeptonPairMuons": np.array(ak.fill_none(abs(lepton1.pdgId) == 13, False)),
                "IsLeptonPairElectron": np.array(ak.fill_none(abs(lepton1.pdgId) == 11, False))
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


