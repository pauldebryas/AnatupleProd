import numpy as np
import awkward as ak
from coffea import processor

from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_common, save_anatuple_lepton, save_anatuple_tau, save_Event_bjets
from CoffeaAnalysis.HNLAnalysis.correction_helpers import compute_sf_e, compute_sf_tau, get_trigger_correction_tau
from CoffeaAnalysis.HNLAnalysis.helpers import IsoElectron_mask, FinalTau_sel, delta_r, bjet_candidates
from CoffeaAnalysis.HNLAnalysis.HNLProcessor import HNLProcessor

class HNLAnalysis_tte(processor.ProcessorABC, HNLProcessor):
    def __init__(self, stitched_list, tag, xsecs):
        HNLProcessor.__init__(self, stitched_list, tag, xsecs)
        acc_dict = {}
        self.selections = self.get_selections()
        for selection in self.selections:
            acc_dict[f'n_ev_{selection}'] = processor.defaultdict_accumulator(int)
            acc_dict[f'sumw_{selection}'] = processor.defaultdict_accumulator(float)
        self._accumulator = processor.dict_accumulator(acc_dict)

        #the corresponding data sample for tte channel (HLT=DoubleTau)
        self.dataHLT = 'Tau_2018'

    @property
    def accumulator(self):
        return self._accumulator

    @staticmethod
    def get_selections():
        return [
            'init',
            'reweight',
            'MET_Filter',
            'AtLeast2tau1e',
            'NoAdditionalIsoElectron',
            'noIsoMuon',
            'HLT',
            'eSelection',
            'drTau_e',
            'drTau2Leptons',
            'corrections'
        ]

    # we will receive a NanoEvents
    def process(self, events):

        out = self.accumulator.identity()
        events, out, ds, mode = self.init_process(out, events)

        # cut specific for that channel 
        self.cut_mu_iso = 0.15 # veto events with tight mu isolation for ortogonality in signal region for channels with muon
        self.cut_tau_idVSe = 6 # require tight isolation against electron for channel with reco electron
        # trigger requirement 
        self.cut_tau_pt = 40.  
        self.cut_tau_eta = 2.1 

        events = self.Lepton_selection(events, mode)

        self.analyse_tte(events, out, ds, mode)

        return out

    def analyse_tte(self, events, out, ds, mode):

        # select tte events: require at least 2 reco tau and 1 reco e 
        events_tte = events[(ak.num(events.SelElectron) >= 1) & (ak.num(events.SelTau) >= 2)]

        out[f'sumw_AtLeast2tau1e'][ds] += ak.sum(events_tte.genWeight)
        out[f'n_ev_AtLeast2tau1e'][ds] += len(events_tte)
        
        # veto events with more than one electron with pfRelIso03_all <= 0.15
        events_tte = events_tte[ IsoElectron_mask(events_tte, 1, iso_cut=0.15)]

        out[f'sumw_NoAdditionalIsoElectron'][ds] += ak.sum(events_tte.genWeight)
        out[f'n_ev_NoAdditionalIsoElectron'][ds] += len(events_tte)

        # save info if an extra electron exist (with 0.15 < pfRelIso03_all < 0.4 )
        events_tte['nAdditionalElectron'] = ak.num(events_tte.SelElectron[events_tte.SelElectron.pfRelIso03_all > 0.15])

        # remove events with more than one isolated muon (assure orthogonality with ttm tmm tem)
        events_tte = events_tte[ak.num(events_tte.SelMuon) == 0]

        out[f'sumw_noIsoMuon'][ds] += ak.sum(events_tte.genWeight)
        out[f'n_ev_noIsoMuon'][ds] += len(events_tte)

        # events should pass most efficient HLT (DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg)
        events_tte = events_tte[events_tte.HLT.DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg]

        out[f'sumw_HLT'][ds] += ak.sum(events_tte.genWeight)
        out[f'n_ev_HLT'][ds] += len(events_tte) 

        #find electron with minimum pfRelIso03_all, in case same isolation, we take the first one by default
        Sel_Electron = events_tte.SelElectron
        
        #select the one with min isolation
        Sel_Electron = Sel_Electron[ak.min(Sel_Electron.pfRelIso03_all, axis=-1) == Sel_Electron.pfRelIso03_all]
        Sel_Electron = Sel_Electron[:,0]

        out[f'sumw_eSelection'][ds] += ak.sum(events_tte.genWeight)
        out[f'n_ev_eSelection'][ds] += len(events_tte) 

        delta_r_cut = 0.5

        # select Tau candidate with dr(electron,Tau)>0.5 
        Tau_candidate = events_tte.SelTau[(delta_r(Sel_Electron,events_tte.SelTau)> delta_r_cut)]

        #make sure at least 1 satisfy this condition
        events_tte = events_tte[ak.num(Tau_candidate) >= 1]
        Sel_Electron = Sel_Electron[ak.num(Tau_candidate) >= 1]
        Tau_candidate = Tau_candidate[ak.num(Tau_candidate) >= 1]

        #Tau1 is the one with the highest isolation VSJet
        Sel_Tau1 = Tau_candidate[ak.max(Tau_candidate.rawDeepTau2018v2p5VSjet, axis=-1) == Tau_candidate.rawDeepTau2018v2p5VSjet]
        Sel_Tau1 = Sel_Tau1[:,0]

        out[f'sumw_drTau_e'][ds] += ak.sum(events_tte.genWeight)
        out[f'n_ev_drTau_e'][ds] += len(events_tte)

        # select Tau2 candidate with dr(e,Tau)>0.5 and dr(Tau1,Tau2)>0.5  and highest isolation VSJet
        events_tte, Sel_Electron, Sel_Tau1, Sel_Tau2 = FinalTau_sel(events_tte, Sel_Electron, Sel_Tau1)

        out[f'sumw_drTau2Leptons'][ds] += ak.sum(events_tte.genWeight)
        out[f'n_ev_drTau2Leptons'][ds] += len(events_tte)
        
        # !!!! must change tau1 and tau2 selection, requiring matching with trigger object beforhand

        bjet_candidate = bjet_candidates(events_tte, Sel_Electron, Sel_Tau1, Sel_Tau2)

        events_tte['nbjets'] = ak.num(bjet_candidate)
        events_tte['bjets'] = bjet_candidate

        if len(events_tte) == 0:
            print('0 events pass selection')
            return
        
        events_tte['matchingHLTDoubleTau'] = self.matching_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_old(events_tte, Sel_Tau1, Sel_Tau2)

        #computing corrections
        if mode != 'Data':
            sf_tau1 = compute_sf_tau(Sel_Tau1)
            sf_tau2 = compute_sf_tau(Sel_Tau2)
            sf_e = compute_sf_e(Sel_Electron)
            Trigger_eff_corr_tau1 = get_trigger_correction_tau(Sel_Tau1)
            Trigger_eff_corr_tau2 = get_trigger_correction_tau(Sel_Tau2)
            events_tte.genWeight = events_tte.genWeight * sf_tau1 * sf_tau2 * sf_e * Trigger_eff_corr_tau1 * Trigger_eff_corr_tau2

        out[f'sumw_corrections'][ds] += ak.sum(events_tte.genWeight)
        out[f'n_ev_corrections'][ds] += len(events_tte)

        # Save anatuple
        self.save_anatuple_tte(ds, events_tte, Sel_Electron, Sel_Tau1, Sel_Tau2, self.tag, mode)

        return events_tte

    def matching_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_old(self, events, Sel_Tau1, Sel_Tau2, min_dr_cut=0.2):

        # select reco tau that match HLT
        # Trigger tau is a tau (id == 15),abs(eta) < 2.2 and pt > 0 (HLT) with  TrigObj_filterBits for Tau: 2 = MediumChargedIso, 16 = HPS, 64 = di-tau
        Trigger_Tau = events.TrigObj[ (abs(events.TrigObj.id) == 15) & (abs(events.TrigObj.eta) < 2.2) & ((events.TrigObj.filterBits & (2+16+64)) != 0)] 

        cut_dr_mask_tau1 = delta_r(Sel_Tau1, Trigger_Tau) < min_dr_cut # remove too high dr matching
        cut_dr_mask_tau2 = delta_r(Sel_Tau2, Trigger_Tau) < min_dr_cut # remove too high dr matching
    
        return (ak.num(cut_dr_mask_tau1) >= 1) & (ak.num(cut_dr_mask_tau2) >= 1) 

    def save_anatuple_tte(self, ds, events, Sel_Electron, Sel_Tau1, Sel_Tau2, tag, mode):

        exclude_list = ['genPartIdx']
        
        save_file, lst = save_anatuple_common(ds, events, tag)

        #info specific to the channel
        lst["nAdditionalElectron"] = np.array(events.nAdditionalElectron)
        lst["matchingHLTDoubleTau"] = np.array(events.matchingHLTDoubleTau)
        lst['channelIndex'] = np.ones(len(events))*331

        if mode == 'signal':
            if 'HNL_tau_M-' not in ds:
                raise 'signal samples not starting with HNL_tau_M-'
            lst['HNLmass'] = np.ones(len(events))*int(ds[len('HNL_tau_M-'):])

        lst = save_anatuple_lepton(Sel_Electron, lst, exclude_list, 'Electron')
        lst = save_anatuple_tau(events, Sel_Tau1, lst, exclude_list, mode, 'Tau1')
        lst = save_anatuple_tau(events, Sel_Tau2, lst, exclude_list, mode, 'Tau2')

        save_Event_bjets(save_file, lst, events)

        return

    def postprocess(self, accumulator):
        return accumulator