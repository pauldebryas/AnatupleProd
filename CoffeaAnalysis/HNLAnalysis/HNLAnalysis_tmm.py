import numpy as np
import awkward as ak
from coffea import processor
import uproot

from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_common, save_anatuple_lepton, save_anatuple_tau, save_Event_bjets
from CoffeaAnalysis.HNLAnalysis.helpers import apply_golden_run, apply_reweight, apply_MET_Filter
from CoffeaAnalysis.HNLAnalysis.correction_helpers import get_pileup_correction, compute_sf_mu, compute_sf_tau_e, get_trigger_correction_mu
from CoffeaAnalysis.HNLAnalysis.helpers import matching_IsoMu24, IsoMuon_mask, Trigger_Muon_sel, FinalTau_sel, delta_r, bjet_candidates

class HNLAnalysis_tmm(processor.ProcessorABC):
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
        cut_tau_idVSe = 2 # idDeepTau2018v2p5VSe >= VVLoose
        cut_tau_idVSjet = 2 # idDeepTau2018v2p5VSjet >= VVLoose

        #muons
        cut_mu_pt = 10. # Muon_pt > cut_mu_pt
        cut_mu_eta = 2.4 # abs(Muon_eta) < cut_mu_eta
        cut_mu_dz = 0.2 #abs(Muon_dz) < cut_mu_dz
        cut_mu_dxy = 0.045 # abs(Muon_dxy) < cut_mu_dxy
        cut_mu_iso = 0.4 # Muon_pfRelIso03_all < cut_mu_iso

        #electrons
        cut_e_pt = 10. # Electron_pt > cut_e_pt
        cut_e_eta = 2.5 # abs(Electron_eta) < cut_e_eta
        cut_e_dz = 0.2 #abs(Electron_dz) < cut_e_dz
        cut_e_dxy = 0.045 # abs(Electron_dxy) < cut_e_dxy
        cut_e_iso = 0.15 # Electron_pfRelIso03_all < cut_e_iso

        #tau
        # + remove decay mode 5 and 6 as suggested here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
        events['SelTau'] = events.Tau[(events.Tau.pt > cut_tau_pt) & (np.abs(events.Tau.eta) < cut_tau_eta) & (np.abs(events.Tau.dz) < cut_tau_dz) & (events.Tau.idDeepTau2018v2p5VSmu >= cut_tau_idVSmu) & (events.Tau.idDeepTau2018v2p5VSe >= cut_tau_idVSe) & (events.Tau.idDeepTau2018v2p5VSjet >= cut_tau_idVSjet) & (events.Tau.decayMode != 5) & (events.Tau.decayMode != 6)]

        #muons
        # + Muon_mediumId
        events['SelMuon'] = events.Muon[(events.Muon.pt > cut_mu_pt) & (np.abs(events.Muon.eta) < cut_mu_eta) & (np.abs(events.Muon.dz) < cut_mu_dz) & (np.abs(events.Muon.dxy) < cut_mu_dxy) & (events.Muon.mediumId > 0) & (events.Muon.pfRelIso03_all < cut_mu_iso)]

        #electrons
        # + mvaNoIso_WP90 > 0 (i.e True)
        events['SelElectron'] = events.Electron[(events.Electron.pt > cut_e_pt) & (np.abs(events.Electron.eta) < cut_e_eta) & (np.abs(events.Electron.dz) < cut_e_dz) & (np.abs(events.Electron.dxy) < cut_e_dxy) & (events.Electron.mvaNoIso_WP90 > 0) & (events.Electron.pfRelIso03_all < cut_e_iso)]

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

        # save info if an extra muon exist (with 0.15 <pfRelIso03_all < 0.4)
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
        Mu2_candidate = Mu2_candidate[Mu2_candidate.pt >= 15]

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
            return

        events_tmm['matchingIsoMu24'] = (matching_IsoMu24(events_tmm, Sel_Muon) | matching_IsoMu24(events_tmm, Sel_Muon2))

        #computing corrections
        if mode != 'Data':
            pileup_corr = get_pileup_correction(events_tmm, 'nominal')* self.norm_factor
            sf_mu1 = compute_sf_mu(Sel_Muon)
            Trigger_eff_corr_mu = get_trigger_correction_mu(Sel_Muon)
            sf_mu2 = compute_sf_mu(Sel_Muon2)
            # as we use DeepTau2018v2p5, we don't have corrections yet: only tau energy sf
            sf_tau = compute_sf_tau_e(Sel_Tau)
            events_tmm.genWeight = events_tmm.genWeight * sf_mu1 * Trigger_eff_corr_mu * sf_mu2 * sf_tau * pileup_corr

        out[f'sumw_corrections'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_corrections'][ds] += len(events_tmm)

        # Save anatuple
        self.save_anatuple_tmm(ds, events_tmm, Sel_Muon, Sel_Muon2, Sel_Tau, self.tag, mode)

        return events_tmm
    
    def save_anatuple_tmm(self, ds, events, Sel_Muon, Sel_Muon2, Sel_Tau, tag, mode):

        exclude_list = ['jetIdxG','genPartIdxG']
        
        save_file, lst = save_anatuple_common(ds, events, tag)

        #info specific to the channel
        lst["nAdditionalMuon"] = np.array(events.nAdditionalMuon)
        lst["matchingIsoMu24"] = np.array(events.matchingIsoMu24)

        lst = save_anatuple_lepton(Sel_Muon, lst, exclude_list, 'Muon1')
        lst = save_anatuple_lepton(Sel_Muon2, lst, exclude_list, 'Muon2')
        lst = save_anatuple_tau(events, Sel_Tau, lst, exclude_list, mode, 'Tau')

        save_Event_bjets(save_file, lst, events)

        return
    
    def postprocess(self, accumulator):
        return accumulator