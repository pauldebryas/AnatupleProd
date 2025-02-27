import numpy as np
import awkward as ak
from coffea import processor
from collections import defaultdict
import copy
import os

from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_lepton, save_Event, add_gen_matching_info
from CoffeaAnalysis.HNLAnalysis.helpers import ll_from_Z_sel, FinalLL_sel
from CoffeaAnalysis.HNLAnalysis.HNLProcessor import HNLProcessor

class HNLAnalysis_Zmu(processor.ProcessorABC, HNLProcessor):
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
            'llinZ',
            'muSel'
        ]

    # we will receive a NanoEvents
    def process(self, events):

        out = copy.deepcopy(self._accumulator)
        events, out = self.init_process(out, events)

        # Do the general lepton selection
        events_Zmu = self.Lepton_selection(events)

        # Apply the cuts and select leptons
        events_Zmu, lepton1, lepton2, Sel_Mu = self.analyse_Zmu(events_Zmu, out)

        # save GenPart info in case MC sample
        if self.mode != 'Data':
            lepton1 = add_gen_matching_info(events_Zmu, lepton1)
            lepton2 = add_gen_matching_info(events_Zmu, lepton2)
            Sel_Mu  = add_gen_matching_info(events_Zmu, Sel_Mu)

        # Save anatuple
        print(f'Saving {len(events_Zmu)} events')
        save_file, lst = self.save_anatuple_Zmu(events_Zmu, lepton1, lepton2, Sel_Mu, self.tag)
        
        save_Event(save_file, lst, 'Events')

        return out

    def analyse_Zmu(self, events, out):
        # l1 and l2 should be 2 mu or 2 e with OS and m(l1,l2) ~ m_Z
        # l3 is a mu 

        # select lll events: require at least 3 reco mu or 2 reco e and 1 reco mu
        events_Zmu = events[((ak.num(events.SelElectron) >= 2) & (ak.num(events.SelMuon) >= 1)) | (ak.num(events.SelMuon) >= 3)]

        out[f'sumw_lllsel'][self.ds] += ak.sum(events_Zmu.genWeight)
        out[f'n_ev_lllsel'][self.ds] += len(events_Zmu)

        '''
        Filters events with two leptons (muons or electrons) satisfying:
            - Invariant mass of 91.2 ± 15 GeV
            - Opposite charges
        Selects the pair with invariant mass closest to the Z mass (91.2 GeV).
        '''
        events_Zmu, lepton1, lepton2 = ll_from_Z_sel(events_Zmu)

        out[f'sumw_llinZ'][self.ds] += ak.sum(events_Zmu.genWeight)
        out[f'n_ev_llinZ'][self.ds] += len(events_Zmu)

        # select Mu candidate with dr(l1,Muon)>0.5 and dr(l2,Muon)>0.5 and minimum pfRelIso03_all
        events_Zmu, lepton1, lepton2, Sel_Mu = FinalLL_sel(events_Zmu, lepton1, lepton2, 'muon')

        out[f'sumw_muSel'][self.ds] += ak.sum(events_Zmu.genWeight)
        out[f'n_ev_muSel'][self.ds] += len(events_Zmu)

        return events_Zmu, lepton1, lepton2, Sel_Mu
    
    def save_anatuple_Zmu(self, events, lepton1, lepton2, SelMuon, tag):

        path = f'/afs/cern.ch/work/p/pdebryas/HNL/tmp/{self.period}/{tag}/Zmu/{self.sample_name}/'

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
                "IsLeptonPairMuons": np.array(ak.fill_none(abs(lepton1.pdgId) == 13, False)),
                "IsLeptonPairElectron": np.array(ak.fill_none(abs(lepton1.pdgId) == 11, False))
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

