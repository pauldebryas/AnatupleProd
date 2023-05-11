import numpy as np
import awkward as ak
from coffea import processor

from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_common, save_anatuple_lepton, save_anatuple_tau, save_Event_bjets
from CoffeaAnalysis.HNLAnalysis.correction_helpers import compute_sf_mu, compute_sf_tau, get_trigger_correction_mu
from CoffeaAnalysis.HNLAnalysis.helpers import IsoMuon_mask, Trigger_Muon_sel, FinalTau_sel, delta_r, bjet_candidates
from CoffeaAnalysis.HNLAnalysis.HNLProcessor import HNLProcessor

class HNLAnalysis_tmm(processor.ProcessorABC, HNLProcessor):
    def __init__(self, stitched_list, tag, xsecs):
        HNLProcessor.__init__(self, stitched_list, tag, xsecs)
        acc_dict = {}
        self.selections = self.get_selections()
        for selection in self.selections:
            acc_dict[f'n_ev_{selection}'] = processor.defaultdict_accumulator(int)
            acc_dict[f'sumw_{selection}'] = processor.defaultdict_accumulator(float)
        self._accumulator = processor.dict_accumulator(acc_dict)

        #the corresponding data sample for tmm channel (HLT=IsoMu24)
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
            'AtLeast1tau2mu',
            'NoAdditionalIsoMuon',
            'NoIsoElectron',
            'HLT',
            'MuTriggerMatching',
            'drMu1Mu2',
            'drTauMuons',
            'corrections']

    # we will receive a NanoEvents
    def process(self, events):

        out = self.accumulator.identity()
        events, out, ds, mode = self.init_process(out, events)
        
        # cut specific for that channel
        self.cut_e_iso = 0.15 # veto events with tight e isolation for ortogonality in signal region for channels with electrons

        events = self.Lepton_selection(events, mode)

        self.analyse_tmm(events, out, ds, mode)

        return out

    def analyse_tmm(self, events, out, ds, mode):

        # select tmm events: require at least 2 reco mu and 1 reco tau 
        events_tmm = events[(ak.num(events.SelMuon) >= 2) & (ak.num(events.SelTau) >= 1)]

        out[f'sumw_AtLeast1tau2mu'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_AtLeast1tau2mu'][ds] += len(events_tmm)

        # veto events with more than two muon with pfRelIso03_all <= 0.15
        events_tmm = events_tmm[IsoMuon_mask(events_tmm, 2, iso_cut=0.15)]

        out[f'sumw_NoAdditionalIsoMuon'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_NoAdditionalIsoMuon'][ds] += len(events_tmm)

        # save info if an extra muon exist with 0.15 <pfRelIso03_all < 0.4
        events_tmm['nAdditionalMuon'] = ak.num(events_tmm.SelMuon[events_tmm.SelMuon.pfRelIso03_all > 0.15])

        # remove events with more than one isolated electron (assure orthogonality with tte tee tem)
        events_tmm = events_tmm[ak.num(events_tmm.SelElectron) == 0]

        out[f'sumw_NoIsoElectron'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_NoIsoElectron'][ds] += len(events_tmm)

        # events should pass most efficient HLT (for now only 1: IsoMu24)
        events_tmm = events_tmm[events_tmm.HLT.IsoMu24]

        out[f'sumw_HLT'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_HLT'][ds] += len(events_tmm)

        events_tmm, Sel_Muon = Trigger_Muon_sel(events_tmm)

        out[f'sumw_MuTriggerMatching'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_MuTriggerMatching'][ds] += len(events_tmm)

        delta_r_cut = 0.5

        # select Muon2, that should be away from Muon1:  dr(Mu2,Mu1)>0.5 & pt>15GeV
        Mu2_candidate = events_tmm.SelMuon[(delta_r(Sel_Muon,events_tmm.SelMuon) > delta_r_cut)]

        #make sure at least 1 satisfy this condition
        events_tmm = events_tmm[ak.num(Mu2_candidate) >= 1]
        Sel_Muon = Sel_Muon[ak.num(Mu2_candidate) >= 1]
        Mu2_candidate = Mu2_candidate[ak.num(Mu2_candidate) >= 1]
        
        #Mu2 is the one with the minimum pfRelIso03_all
        Sel_Muon2 = Mu2_candidate[ak.min(Mu2_candidate.pfRelIso03_all, axis=-1) == Mu2_candidate.pfRelIso03_all]
        Sel_Muon2 = Sel_Muon2[:,0]

        out[f'sumw_drMu1Mu2'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_drMu1Mu2'][ds] += len(events_tmm)

        # select Tau candidate with dr(muon,Tau)>0.5 and dr(muon2,Tau)>0.5  and highest isolation VSJet
        events_tmm, Sel_Muon, Sel_Muon2, Sel_Tau = FinalTau_sel(events_tmm, Sel_Muon, Sel_Muon2)

        out[f'sumw_drTauMuons'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_drTauMuons'][ds] += len(events_tmm)

        bjet_candidate = bjet_candidates(events_tmm, Sel_Muon, Sel_Muon2, Sel_Tau)

        events_tmm['nbjets'] = ak.num(bjet_candidate)
        events_tmm['bjets'] = bjet_candidate

        if len(events_tmm) == 0:
            print('0 events pass selection')
            return
 
        #computing sf corrections
        if mode != 'Data':
            sf_tau = compute_sf_tau(Sel_Tau)
            sf_mu1 = compute_sf_mu(Sel_Muon)
            sf_mu2 = compute_sf_mu(Sel_Muon2)
            Trigger_eff_corr_mu = get_trigger_correction_mu(Sel_Muon) # apply Trigger sf to the Muon that match HLT
            events_tmm.genWeight = events_tmm.genWeight * sf_tau * sf_mu1 * sf_mu2 * Trigger_eff_corr_mu 

        out[f'sumw_corrections'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_corrections'][ds] += len(events_tmm)

        # Save anatuple
        self.save_anatuple_tmm(ds, events_tmm, Sel_Muon, Sel_Muon2, Sel_Tau, self.tag, mode)

        return events_tmm
    
    def save_anatuple_tmm(self, ds, events, Sel_Muon, Sel_Muon2, Sel_Tau, tag, mode):

        exclude_list = ['genPartIdx']
        
        save_file, lst = save_anatuple_common(ds, events, tag)

        #info specific to the channel
        lst["nAdditionalMuon"] = np.array(events.nAdditionalMuon)
        lst['channelIndex'] = np.ones(len(events))*322

        if mode == 'signal':
            if 'HNL_tau_M-' not in ds:
                raise 'signal samples not starting with HNL_tau_M-'
            lst['HNLmass'] = np.ones(len(events))*int(ds[len('HNL_tau_M-'):])

        lst = save_anatuple_lepton(Sel_Muon, lst, exclude_list, 'Muon1')
        lst = save_anatuple_lepton(Sel_Muon2, lst, exclude_list, 'Muon2')
        lst = save_anatuple_tau(events, Sel_Tau, lst, exclude_list, mode, 'Tau')

        save_Event_bjets(save_file, lst, events)

        return
    
    def postprocess(self, accumulator):
        return accumulator