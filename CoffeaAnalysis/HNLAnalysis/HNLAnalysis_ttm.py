import numpy as np
import awkward as ak
from coffea import processor

from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_common, save_anatuple_lepton, save_anatuple_tau, save_Event_bjets
from CoffeaAnalysis.HNLAnalysis.correction_helpers import compute_sf_mu, compute_sf_tau, get_trigger_correction_mu
from CoffeaAnalysis.HNLAnalysis.helpers import IsoMuon_mask, Trigger_Muon_sel, FinalTau_sel, delta_r, bjet_candidates
from CoffeaAnalysis.HNLAnalysis.HNLProcessor import HNLProcessor

class HNLAnalysis_ttm(processor.ProcessorABC, HNLProcessor):
    def __init__(self, stitched_list, tag, xsecs):
        HNLProcessor.__init__(self, stitched_list, tag, xsecs)
        acc_dict = {}
        self.selections = self.get_selections()
        for selection in self.selections:
            acc_dict[f'n_ev_{selection}'] = processor.defaultdict_accumulator(int)
            acc_dict[f'sumw_{selection}'] = processor.defaultdict_accumulator(float)
        self._accumulator = processor.dict_accumulator(acc_dict)
        
        #the corresponding data sample for ttm channel (HLT=IsoMu24)
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
            'AtLeast2tau1mu',
            'NoAdditionalIsoMuon',
            'NoIsoElectron',
            'HLT',
            'MuTriggerMatching',
            'drMuTau',
            'drTau2Leptons',
            'corrections'
        ]

    # we will receive a NanoEvents
    def process(self, events):

        out = self.accumulator.identity()
        events, out, ds, mode = self.init_process(out, events)

        # cut specific for that channel
        self.cut_e_iso = 0.15 # veto events with tight e isolation for ortogonality in signal region for channels with electrons

        events = self.Lepton_selection(events, mode)

        self.analyse_ttm(events, out, ds, mode)

        return out

    def analyse_ttm(self, events, out, ds, mode):

        # select ttm events: require at least 2 reco tau and 1 reco mu
        events_ttm = events[(ak.num(events.SelMuon) >= 1) & (ak.num(events.SelTau) >= 2)]

        out[f'sumw_AtLeast2tau1mu'][ds] += ak.sum(events_ttm.genWeight)
        out[f'n_ev_AtLeast2tau1mu'][ds] += len(events_ttm)
        
        # remove events with more than one muon with pfRelIso03_all <= 0.15
        events_ttm = events_ttm[IsoMuon_mask(events_ttm, 1, iso_cut=0.15)]

        out[f'sumw_NoAdditionalIsoMuon'][ds] += ak.sum(events_ttm.genWeight)
        out[f'n_ev_NoAdditionalIsoMuon'][ds] += len(events_ttm)

        # save info if an extra muon exist (with 0.15 <pfRelIso03_all < 0.4)
        events_ttm['nAdditionalMuon'] = ak.num(events_ttm.SelMuon[events_ttm.SelMuon.pfRelIso03_all > 0.15])

        # remove events with more than one isolated electron (assure orthogonality with tte tee tem)
        events_ttm = events_ttm[ak.num(events_ttm.SelElectron) == 0]

        out[f'sumw_NoIsoElectron'][ds] += ak.sum(events_ttm.genWeight)
        out[f'n_ev_NoIsoElectron'][ds] += len(events_ttm)

        # events should pass most efficient HLT (for now only 1: IsoMu24)
        events_ttm = events_ttm[events_ttm.HLT.IsoMu24]

        out[f'sumw_HLT'][ds] += ak.sum(events_ttm.genWeight)
        out[f'n_ev_HLT'][ds] += len(events_ttm)  

        events_ttm, Sel_Muon = Trigger_Muon_sel(events_ttm)

        out[f'sumw_MuTriggerMatching'][ds] += ak.sum(events_ttm.genWeight)
        out[f'n_ev_MuTriggerMatching'][ds] += len(events_ttm)  

        delta_r_cut = 0.5

        # select Tau candidate with dr(muon,Tau)>0.5 
        Tau1_candidate = events_ttm.SelTau[(delta_r(Sel_Muon,events_ttm.SelTau)> delta_r_cut)]

        #make sure at least 1 satisfy this condition
        events_ttm = events_ttm[ak.num(Tau1_candidate) >= 1]
        Sel_Muon = Sel_Muon[ak.num(Tau1_candidate) >= 1]
        Tau1_candidate = Tau1_candidate[ak.num(Tau1_candidate) >= 1]

        #Tau1 is the one with the highest isolation VSJet
        Sel_Tau1 = Tau1_candidate[ak.max(Tau1_candidate.rawDeepTau2018v2p5VSjet, axis=-1) == Tau1_candidate.rawDeepTau2018v2p5VSjet]
        Sel_Tau1 = Sel_Tau1[:,0]

        out[f'sumw_drMuTau'][ds] += ak.sum(events_ttm.genWeight)
        out[f'n_ev_drMuTau'][ds] += len(events_ttm)

        # select Tau candidate with dr(muon,Tau)>0.5 and dr(Tau1,Tau)>0.5  and highest isolation VSJet
        events_ttm, Sel_Muon, Sel_Tau1, Sel_Tau2 = FinalTau_sel(events_ttm, Sel_Muon, Sel_Tau1)

        out[f'sumw_drTau2Leptons'][ds] += ak.sum(events_ttm.genWeight)
        out[f'n_ev_drTau2Leptons'][ds] += len(events_ttm)
        
        bjet_candidate = bjet_candidates(events_ttm, Sel_Muon, Sel_Tau1, Sel_Tau2)

        events_ttm['nbjets'] = ak.num(bjet_candidate)
        events_ttm['bjets'] = bjet_candidate

        if len(events_ttm) == 0:
            print('0 events pass selection')
            return

        #computing corrections
        if mode != 'Data':
            sf_tau1 = compute_sf_tau(Sel_Tau1)
            sf_tau2 = compute_sf_tau(Sel_Tau2)
            sf_mu = compute_sf_mu(Sel_Muon)
            Trigger_eff_corr_mu = get_trigger_correction_mu(Sel_Muon)
            events_ttm.genWeight = events_ttm.genWeight * sf_tau1 * sf_tau2 * sf_mu * Trigger_eff_corr_mu

        out[f'sumw_corrections'][ds] += ak.sum(events_ttm.genWeight)
        out[f'n_ev_corrections'][ds] += len(events_ttm)

        # Save anatuple
        self.save_anatuple_ttm(ds, events_ttm, Sel_Muon, Sel_Tau1, Sel_Tau2, self.tag, mode)

        return events_ttm

    def save_anatuple_ttm(self, ds, events, Sel_Muon, Sel_Tau1, Sel_Tau2, tag, mode):

        exclude_list = ['genPartIdx']
        
        save_file, lst = save_anatuple_common(ds, events, tag)

        #info specific to the channel
        lst["nAdditionalMuon"] = np.array(events.nAdditionalMuon)
        lst['channelIndex'] = np.ones(len(events))*332

        if mode == 'signal':
            if 'HNL_tau_M-' not in ds:
                raise 'signal samples not starting with HNL_tau_M-'
            lst['HNLmass'] = np.ones(len(events))*int(ds[len('HNL_tau_M-'):])

        lst = save_anatuple_lepton(Sel_Muon, lst, exclude_list, 'Muon')
        lst = save_anatuple_tau(events, Sel_Tau1, lst, exclude_list, mode, 'Tau1')
        lst = save_anatuple_tau(events, Sel_Tau2, lst, exclude_list, mode, 'Tau2')

        save_Event_bjets(save_file, lst, events)
        
        return

    def postprocess(self, accumulator):
        return accumulator