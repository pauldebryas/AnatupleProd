import numpy as np
import awkward as ak
from coffea import processor
import uproot

from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_common, save_anatuple_lepton, save_anatuple_tau, save_Event_bjets
from CoffeaAnalysis.HNLAnalysis.helpers import apply_golden_run, apply_reweight, apply_MET_Filter
from CoffeaAnalysis.HNLAnalysis.correction_helpers import get_pileup_correction, compute_sf_e, compute_sf_tau_e, get_trigger_correction_tau
from CoffeaAnalysis.HNLAnalysis.helpers import IsoElectron_mask, FinalTau_sel, delta_r, bjet_candidates

class HNLAnalysis_tte(processor.ProcessorABC):
    def __init__(self, stitched_list, tag, xsecs):
        acc_dict = {}
        self.selections = self.get_selections()
        for selection in self.selections:
            acc_dict[f'n_ev_{selection}'] = processor.defaultdict_accumulator(int)
            acc_dict[f'sumw_{selection}'] = processor.defaultdict_accumulator(float)
        self._accumulator = processor.dict_accumulator(acc_dict)

        if stitched_list is None or len(stitched_list) == 0:
            raise 'Missing stitched_list in samples_2018.yaml'
        self.stitched_list = stitched_list
        if tag is None or len(tag) == 0:
            raise 'Missing tag'
        self.tag = tag
        if xsecs is None:
            raise 'Missing xsecs'
        self.xsecs = xsecs
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

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        ds = events.metadata["dataset"] # dataset name
        print('Processing: ' + ds)

        # defining the mode
        mode ='MCbackground' # default mode

        if self.dataHLT in ds:
            mode ='Data'
            # only keep the "golden" runs
            events = apply_golden_run(ds, events)
            #A,B,C and D together
            ds = ds[0:-1]
        
        if 'HNL' in ds:
            mode ='signal'

        out['sumw_init'][ds] += ak.sum(events.genWeight)
        out['n_ev_init'][ds] += len(events)

        if mode != 'Data':
            #reweights events with lumi x xsec / n_events + applying stitching weights for DY and WJets samples
            events = apply_reweight(ds, events, self.stitched_list, self.dataHLT, self.xsecs)
            # pileup correction: compute normalizing factor in order to keep the same number of events before and after correction (before any cut)
            corr = get_pileup_correction(events, 'nominal')
            self.norm_factor = ak.sum(events.genWeight)/ak.sum(events.genWeight*corr)
        
        out['sumw_reweight'][ds] += ak.sum(events.genWeight)
        out['n_ev_reweight'][ds] += len(events)

        # MET filters
        events = apply_MET_Filter(events)

        out['sumw_MET_Filter'][ds] += ak.sum(events.genWeight)
        out['n_ev_MET_Filter'][ds] += len(events)

        # Reco event selection: common minimal requirement for leptons
        #tau
        cut_tau_pt = 40. # Tau_pt > cut_tau_pt
        cut_tau_eta = 2.1 #abs(Tau_eta) < cut_tau_eta
        cut_tau_dz = 0.2 #abs(Tau_dz) < cut_tau_dz
        cut_tau_idVSmu = 4 # idDeepTau2018v2p5VSmu >= Tight
        cut_tau_idVSe = 6 # idDeepTau2018v2p5VSe >= Tight
        cut_tau_idVSjet = 2 # idDeepTau2018v2p5VSjet >= VVLoose
    
        #electrons
        cut_e_pt = 10. # Electron_pt > cut_e_pt
        cut_e_eta = 2.5 # abs(Electron_eta) < cut_e_eta
        cut_e_dz = 0.2 #abs(Electron_dz) < cut_e_dz
        cut_e_dxy = 0.045 # abs(Electron_dxy) < cut_e_dxy
        cut_e_iso = 0.4 # Electron_pfRelIso03_all < cut_e_iso

        #muons
        cut_mu_pt = 10. # Muon_pt > cut_mu_pt
        cut_mu_eta = 2.4 # abs(Muon_eta) < cut_mu_eta
        cut_mu_dz = 0.2 #abs(Muon_dz) < cut_mu_dz
        cut_mu_dxy = 0.045 # abs(Muon_dxy) < cut_mu_dxy
        cut_mu_iso = 0.15 # Muon_pfRelIso03_all <= cut_mu_iso

        #tau
        # + remove decay mode 5 and 6 as suggested here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
        events['SelTau'] = events.Tau[(events.Tau.pt > cut_tau_pt) & (np.abs(events.Tau.eta) < cut_tau_eta) & (np.abs(events.Tau.dz) < cut_tau_dz) & (events.Tau.idDeepTau2018v2p5VSmu >= cut_tau_idVSmu) & (events.Tau.idDeepTau2018v2p5VSe >= cut_tau_idVSe) & (events.Tau.idDeepTau2018v2p5VSjet >= cut_tau_idVSjet) & (events.Tau.decayMode != 5) & (events.Tau.decayMode != 6)]

        #electron
        # + mvaNoIso_WP90 > 0 (i.e True)
        events['SelElectron'] = events.Electron[(events.Electron.pt > cut_e_pt) & (np.abs(events.Electron.eta) < cut_e_eta) & (np.abs(events.Electron.dz) < cut_e_dz) & (np.abs(events.Electron.dxy) < cut_e_dxy) & (events.Electron.mvaNoIso_WP90 > 0) & (events.Electron.pfRelIso03_all < cut_e_iso)]

        #muons 
        # + Muon_mediumId
        events['SelMuon'] = events.Muon[(events.Muon.pt > cut_mu_pt) & (np.abs(events.Muon.eta) < cut_mu_eta) & (np.abs(events.Muon.dz) < cut_mu_dz) & (np.abs(events.Muon.dxy) < cut_mu_dxy) & (events.Muon.mediumId > 0) & (events.Muon.pfRelIso03_all <= cut_mu_iso)]

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
        #first make sur e_pt>20
        Sel_Electron = events_tte.SelElectron[events_tte.SelElectron.pt >= 20]

        events_tte = events_tte[ak.num(Sel_Electron) >= 1]
        Sel_Electron = Sel_Electron[ak.num(Sel_Electron) >= 1]
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
        
        bjet_candidate = bjet_candidates(events_tte, Sel_Electron, Sel_Tau1, Sel_Tau2)

        events_tte['nbjets'] = ak.num(bjet_candidate)
        events_tte['bjets'] = bjet_candidate

        if len(events_tte) == 0:
            return
        
        events_tte['matchingHLTDoubleTau'] = self.matching_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_old(events_tte, Sel_Tau1, Sel_Tau2)

        #computing corrections
        if mode != 'Data':
            pileup_corr = get_pileup_correction(events_tte, 'nominal')* self.norm_factor
            sf_e = compute_sf_e(Sel_Electron)
            Trigger_eff_corr_tau = get_trigger_correction_tau(Sel_Tau1)
            # as we use DeepTau2018v2p5, we don't have corrections yet: only tau energy sf
            sf_tau1 = compute_sf_tau_e(Sel_Tau1)
            sf_tau2 = compute_sf_tau_e(Sel_Tau2)
            events_tte.genWeight = events_tte.genWeight * sf_e  * sf_tau1 * sf_tau2 * Trigger_eff_corr_tau * pileup_corr

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

        exclude_list = ['jetIdxG','genPartIdxG']
        
        save_file, lst = save_anatuple_common(ds, events, tag)

        #info specific to the channel
        lst["nAdditionalElectron"] = np.array(events.nAdditionalElectron)
        lst["matchingHLTDoubleTau"] = np.array(events.matchingHLTDoubleTau)

        lst = save_anatuple_lepton(Sel_Electron, lst, exclude_list, 'Electron')
        lst = save_anatuple_tau(events, Sel_Tau1, lst, exclude_list, mode, 'Tau1')
        lst = save_anatuple_tau(events, Sel_Tau2, lst, exclude_list, mode, 'Tau2')

        save_Event_bjets(save_file, lst, events)

        return

    def postprocess(self, accumulator):
        return accumulator