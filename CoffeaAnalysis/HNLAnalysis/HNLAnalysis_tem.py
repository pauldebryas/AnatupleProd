import numpy as np
import awkward as ak
from coffea import processor

from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_common, save_anatuple_lepton, save_anatuple_tau, save_Event_bjets
from CoffeaAnalysis.HNLAnalysis.correction_helpers import compute_sf_tau, compute_sf_mu, compute_sf_e, get_trigger_correction_mu
from CoffeaAnalysis.HNLAnalysis.helpers import IsoMuon_mask, IsoElectron_mask, Trigger_Muon_sel, FinalTau_sel, delta_r, bjet_candidates
from CoffeaAnalysis.HNLAnalysis.HNLProcessor import HNLProcessor

class HNLAnalysis_tem(processor.ProcessorABC, HNLProcessor):
    def __init__(self, stitched_list, tag, xsecs):
        HNLProcessor.__init__(self, stitched_list, tag, xsecs)
        acc_dict = {}
        self.selections = self.get_selections()
        for selection in self.selections:
            acc_dict[f'n_ev_{selection}'] = processor.defaultdict_accumulator(int)
            acc_dict[f'sumw_{selection}'] = processor.defaultdict_accumulator(float)
        self._accumulator = processor.dict_accumulator(acc_dict)

        #the corresponding data sample for tem channel (HLT=IsoMu24)
        self.dataHLT = 'SingleMuon_2018'

    @property
    def accumulator(self):
        return self._accumulator

    @staticmethod
    def get_selections():
        return [
            'init',
            'reweight',
            'MET_Filter',
            'AtLeast1tau1mu1e',
            'NoAdditionalIsoMuon',
            'NoAdditionalElectron',
            'HLT',
            'MuTriggerMatching',
            'eSelection',
            'drTauLeptons',
            'corrections'
        ]

    # we will receive a NanoEvents
    def process(self, events):

        out = self.accumulator.identity()
        events, out, ds, mode = self.init_process(out, events)

        # cut specific for that channel 
        self.cut_tau_idVSe = 6 # require tight isolation against electron for channel with reco electron

        events = self.Lepton_selection(events, mode)
                              
        self.analyse_tem(events, out, ds, mode)

        return out

    def analyse_tem(self, events, out, ds, mode):

        # select tem events: require at least 1 reco mu, 1 reco tau and 1 reco e
        events_tem = events[(ak.num(events.SelElectron) >= 1) & (ak.num(events.SelMuon) >= 1) & (ak.num(events.SelTau) >= 1)]

        out[f'sumw_AtLeast1tau1mu1e'][ds] += ak.sum(events_tem.genWeight)
        out[f'n_ev_AtLeast1tau1mu1e'][ds] += len(events_tem)

        # veto events with more than one muon with pfRelIso03_all <= 0.15
        events_tem = events_tem[IsoMuon_mask(events_tem, 1, iso_cut=0.15)]

        out[f'sumw_NoAdditionalIsoMuon'][ds] += ak.sum(events_tem.genWeight)
        out[f'n_ev_NoAdditionalIsoMuon'][ds] += len(events_tem)

        # veto events with more than one electron with pfRelIso03_all <= 0.15
        events_tem = events_tem[ IsoElectron_mask(events_tem, 1, iso_cut=0.15)]

        out[f'sumw_NoAdditionalElectron'][ds] += ak.sum(events_tem.genWeight)
        out[f'n_ev_NoAdditionalElectron'][ds] += len(events_tem)

        # save info if an extra muon exist (with 0.15 < pfRelIso03_all < 0.4)
        events_tem['nAdditionalMuon'] = ak.num(events_tem.SelMuon[events_tem.SelMuon.pfRelIso03_all > 0.15])

        # events should pass most efficient HLT (for now only 1: IsoMu24)
        events_tem = events_tem[events_tem.HLT.IsoMu24]

        out[f'sumw_HLT'][ds] += ak.sum(events_tem.genWeight)
        out[f'n_ev_HLT'][ds] += len(events_tem)

        events_tem, Sel_Muon = Trigger_Muon_sel(events_tem)

        out[f'sumw_MuTriggerMatching'][ds] += ak.sum(events_tem.genWeight)
        out[f'n_ev_MuTriggerMatching'][ds] += len(events_tem)

        #find electron with minimum pfRelIso03_all, in case same isolation, we take the first one by default
        Sel_Electron = events_tem.SelElectron
        
        #select the one with min isolation
        Sel_Electron = Sel_Electron[ak.min(Sel_Electron.pfRelIso03_all, axis=-1) == Sel_Electron.pfRelIso03_all]
        Sel_Electron = Sel_Electron[:,0]

        out[f'sumw_eSelection'][ds] += ak.sum(events_tem.genWeight)
        out[f'n_ev_eSelection'][ds] += len(events_tem)

        # select Tau candidate with dr(muon,Tau)>0.5 and dr(electron,Tau)>0.5  and highest isolation VSJet
        events_tem, Sel_Electron, Sel_Muon, Sel_Tau = FinalTau_sel(events_tem, Sel_Electron, Sel_Muon)

        out[f'sumw_drTauLeptons'][ds] += ak.sum(events_tem.genWeight)
        out[f'n_ev_drTauLeptons'][ds] += len(events_tem)

        bjet_candidate = bjet_candidates(events_tem, Sel_Electron, Sel_Muon, Sel_Tau)

        events_tem['nbjets'] = ak.num(bjet_candidate)
        events_tem['bjets'] = bjet_candidate

        if len(events_tem) == 0:
            print('0 events pass selection')
            return

        #computing corrections
        if mode != 'Data':
            sf_tau = compute_sf_tau(Sel_Tau)
            sf_e = compute_sf_e(Sel_Electron)
            sf_mu = compute_sf_mu(Sel_Muon)
            Trigger_eff_corr_mu = get_trigger_correction_mu(Sel_Muon)
            events_tem.genWeight = events_tem.genWeight * sf_tau * sf_e * sf_mu * Trigger_eff_corr_mu 

        out[f'sumw_corrections'][ds] += ak.sum(events_tem.genWeight)
        out[f'n_ev_corrections'][ds] += len(events_tem)

        # Save anatuple
        self.save_anatuple_tem(ds, events_tem, Sel_Muon, Sel_Electron, Sel_Tau, self.tag, mode)

        return events_tem
    
    def save_anatuple_tem(self, ds, events, Sel_Muon, Sel_Electron, Sel_Tau, tag, mode):

        exclude_list = ['genPartIdx']
        
        save_file, lst = save_anatuple_common(ds, events, tag)
        
        #info specific to the channel
        lst["nAdditionalMuon"] = np.array(events.nAdditionalMuon)
        lst['channelIndex'] = np.ones(len(events))*312

        if mode == 'signal':
            if 'HNL_tau_M-' not in ds:
                raise 'signal samples not starting with HNL_tau_M-'
            lst['HNLmass'] = np.ones(len(events))*int(ds[len('HNL_tau_M-'):])

        lst = save_anatuple_lepton(Sel_Muon, lst, exclude_list, 'Muon')
        lst = save_anatuple_lepton(Sel_Electron, lst, exclude_list, 'Electron')
        lst = save_anatuple_tau(events, Sel_Tau, lst, exclude_list, mode, 'Tau')

        save_Event_bjets(save_file, lst, events)
        
        return

    def postprocess(self, accumulator):
        return accumulator