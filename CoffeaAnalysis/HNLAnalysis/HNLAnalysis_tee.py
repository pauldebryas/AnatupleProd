import numpy as np
import awkward as ak
from coffea import processor
import uproot

from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_common, save_anatuple_lepton, save_anatuple_tau, save_Event_bjets
from CoffeaAnalysis.HNLAnalysis.helpers import apply_golden_run, apply_reweight, apply_MET_Filter
from CoffeaAnalysis.HNLAnalysis.correction_helpers import get_pileup_correction, compute_sf_e, compute_sf_tau_e, get_trigger_correction_e
from CoffeaAnalysis.HNLAnalysis.helpers import IsoElectron_mask, FinalTau_sel, delta_r, bjet_candidates

class HNLAnalysis_tee(processor.ProcessorABC):
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
        #the corresponding data sample for tee channel (HLT=Ele32)
        self.dataHLT = 'EGamma_2018'

    @property
    def accumulator(self):
        return self._accumulator

    @staticmethod
    def get_selections():
        return [
            'init',
            'reweight',
            'MET_Filter',
            'AtLeast1tau2e',
            'NoAdditionalIsoElectron',
            'noIsoMuon',
            'HLT',
            'eSelection',
            'tausel',
            'e2sel',
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
        cut_tau_pt = 20. # Tau_pt > cut_tau_pt
        cut_tau_eta = 2.5 #abs(Tau_eta) < cut_tau_eta
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

        self.analyse_tee(events, out, ds, mode)

        return out

    def analyse_tee(self, events, out, ds, mode):

        # select tee events: require 2 reco e and 1 reco tau 
        events_tee = events[(ak.num(events.SelElectron) >= 2) & (ak.num(events.SelTau) >= 1)]

        out[f'sumw_AtLeast1tau2e'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_AtLeast1tau2e'][ds] += len(events_tee)

        # veto events with more than two electrons with pfRelIso03_all <= 0.15
        events_tee = events_tee[ IsoElectron_mask(events_tee, 2, iso_cut=0.15)]

        out[f'sumw_NoAdditionalIsoElectron'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_NoAdditionalIsoElectron'][ds] += len(events_tee)

        # save info if an extra electron exist (with 0.15 <pfRelIso03_all < 0.4)
        events_tee['nAdditionalElectron'] = ak.num(events_tee.SelElectron[events_tee.SelMuon.pfRelIso03_all > 0.15])

        # remove events with more than one isolated muon (assure orthogonality with ttm tmm tem)
        events_tee = events_tee[ak.num(events_tee.SelMuon) == 0]

        out[f'sumw_noIsoMuon'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_noIsoMuon'][ds] += len(events_tee)

        # events should pass most efficient HLT (Ele32_WPTight_Gsf_L1DoubleEG)
        events_tee = events_tee[events_tee.HLT.Ele32_WPTight_Gsf_L1DoubleEG]

        out[f'sumw_HLT'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_HLT'][ds] += len(events_tee)

        #find electron with minimum pfRelIso03_all, in case same isolation, we take the first one by default
        Sel_Electron = events_tee.SelElectron[events_tee.SelElectron.pt > 34]

        events_tee = events_tee[(ak.num(Sel_Electron) >= 1)]
        Sel_Electron = Sel_Electron[(ak.num(Sel_Electron) >= 1)]

        Sel_Electron = Sel_Electron[ak.min(Sel_Electron.pfRelIso03_all, axis=-1) == Sel_Electron.pfRelIso03_all]
        Sel_Electron = Sel_Electron[:,0]

        out[f'sumw_eSelection'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_eSelection'][ds] += len(events_tee) 

        # select Electron2, that should be away from first Electron: dr(e2,e)>0.5
        delta_r_cut = 0.5
        e2_candidate = events_tee.SelElectron[(delta_r(Sel_Electron,events_tee.SelElectron)> delta_r_cut)]
        #with min pt > 20
        e2_candidate = e2_candidate[e2_candidate.pt > 20]

        #make sure at least 1 satisfy this condition
        events_tee = events_tee[ak.num(e2_candidate) >= 1]
        Sel_Electron = Sel_Electron[ak.num(e2_candidate) >= 1]
        e2_candidate = e2_candidate[ak.num(e2_candidate) >= 1]

        #e2 is the one with the minimum pfRelIso03_all
        Sel_Electron2 = e2_candidate[ak.min(e2_candidate.pfRelIso03_all, axis=-1) == e2_candidate.pfRelIso03_all]
        Sel_Electron2 = Sel_Electron2[:,0]

        out[f'sumw_e2sel'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_e2sel'][ds] += len(events_tee)

        # select Tau candidate with dr(e1,Tau)>0.5 and dr(e2,Tau)>0.5 and highest isolation VSJet
        events_tee, Sel_Electron, Sel_Electron2, Sel_Tau = FinalTau_sel(events_tee, Sel_Electron, Sel_Electron2)

        out[f'sumw_tausel'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_tausel'][ds] += len(events_tee)
        
        bjet_candidate = bjet_candidates(events_tee, Sel_Electron, Sel_Electron2, Sel_Tau)

        events_tee['nbjets'] = ak.num(bjet_candidate)
        events_tee['bjets'] = bjet_candidate

        if len(events_tee) == 0:
            return

        events_tee['matchingEle32'] = (self.matching_Ele32_WPTight_Gsf_L1DoubleEG(events_tee, Sel_Electron) | self.matching_Ele32_WPTight_Gsf_L1DoubleEG(events_tee, Sel_Electron2))

       #computing corrections
        if mode != 'Data':
            pileup_corr = get_pileup_correction(events_tee, 'nominal')* self.norm_factor
            sf_e1 = compute_sf_e(Sel_Electron)
            Trigger_eff_corr_e1 = get_trigger_correction_e(Sel_Electron)
            sf_e2 = compute_sf_e(Sel_Electron2)
            # as we use DeepTau2018v2p5, we don't have corrections yet: only tau energy sf
            sf_tau = compute_sf_tau_e(Sel_Tau)
            events_tee.genWeight = events_tee.genWeight * sf_e1 * sf_e2 * sf_tau * Trigger_eff_corr_e1 * pileup_corr

        out[f'sumw_corrections'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_corrections'][ds] += len(events_tee)

        # Save anatuple
        self.save_anatuple_tee(ds, events_tee, Sel_Electron, Sel_Electron2, Sel_Tau, self.tag, mode)

        return events_tee

    def matching_Ele32_WPTight_Gsf_L1DoubleEG(self, events, Sel_Electron, min_dr_cut = 0.2):
        # select reco electron that match HLT
        # Trigger e is a e (id == 11) with  TrigObj_filterBits for Electron: 2 = 1e WPTight
        Trigger_Electron = events.TrigObj[ (abs(events.TrigObj.id) == 11) & ((events.TrigObj.filterBits & (2)) != 0)]  

        cut_dr_mask = delta_r(Sel_Electron, Trigger_Electron) < min_dr_cut # remove too high dr matching

        return ak.num(cut_dr_mask) >= 1

    def save_anatuple_tee(self, ds, events, Sel_Electron, Sel_Electron2, Sel_Tau, tag, mode):

        exclude_list = ['jetIdxG','genPartIdxG']
        
        save_file, lst = save_anatuple_common(ds, events, tag)

        #info specific to the channel
        lst["nAdditionalElectron"] = np.array(events.nAdditionalElectron)
        lst["matchingEle32"] = np.array(events.matchingEle32)

        lst = save_anatuple_lepton(Sel_Electron, lst, exclude_list, 'Electron1')
        lst = save_anatuple_lepton(Sel_Electron2, lst, exclude_list, 'Electron2')
        lst = save_anatuple_tau(events, Sel_Tau, lst, exclude_list, mode, 'Tau')

        save_Event_bjets(save_file, lst, events)

        return

    def postprocess(self, accumulator):
        return accumulator
