import numpy as np
import awkward as ak
from coffea import processor
import uproot

from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_common, save_anatuple_lepton, save_anatuple_tau, save_Event_bjets
from CoffeaAnalysis.HNLAnalysis.helpers import apply_golden_run, apply_reweight, apply_MET_Filter
from CoffeaAnalysis.HNLAnalysis.correction_helpers import get_pileup_correction, compute_sf_mu, compute_sf_e, compute_sf_tau_e, get_trigger_correction_mu
from CoffeaAnalysis.HNLAnalysis.helpers import matching_IsoMu24, IsoMuon_mask, Trigger_Muon_sel, FinalTau_sel, delta_r, bjet_candidates

class HNLAnalysis_tem(processor.ProcessorABC):
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
            'HLT',
            'MuTriggerMatching',
            'NoAdditionalElectron',
            'drTauLeptons',
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
        cut_e_iso = 0.15 # Electron_pfRelIso03_all < cut_e_iso

        #muons
        cut_mu_pt = 10. # Muon_pt > cut_mu_pt
        cut_mu_eta = 2.4 # abs(Muon_eta) < cut_mu_eta
        cut_mu_dz = 0.2 #abs(Muon_dz) < cut_mu_dz
        cut_mu_dxy = 0.045 # abs(Muon_dxy) < cut_mu_dxy
        cut_mu_iso = 0.4 # Muon_pfRelIso03_all < cut_mu_iso

        #tau
        # + remove decay mode 5 and 6 as suggested here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
        events['SelTau'] = events.Tau[(events.Tau.pt > cut_tau_pt) & (np.abs(events.Tau.eta) < cut_tau_eta) & (np.abs(events.Tau.dz) < cut_tau_dz) & (events.Tau.idDeepTau2018v2p5VSmu >= cut_tau_idVSmu) & (events.Tau.idDeepTau2018v2p5VSe >= cut_tau_idVSe) & (events.Tau.idDeepTau2018v2p5VSjet >= cut_tau_idVSjet) & (events.Tau.decayMode != 5) & (events.Tau.decayMode != 6)]

        #electrons
        # + mvaNoIso_WP90 > 0 (i.e True)
        events['SelElectron'] = events.Electron[(events.Electron.pt > cut_e_pt) & (np.abs(events.Electron.eta) < cut_e_eta) & (np.abs(events.Electron.dz) < cut_e_dz) & (np.abs(events.Electron.dxy) < cut_e_dxy) & (events.Electron.mvaNoIso_WP90 > 0) & (events.Electron.pfRelIso03_all < cut_e_iso)]

        #muons 
        # + Muon_mediumId 
        events['SelMuon'] = events.Muon[(events.Muon.pt > cut_mu_pt) & (np.abs(events.Muon.eta) < cut_mu_eta) & (np.abs(events.Muon.dz) < cut_mu_dz) & (np.abs(events.Muon.dxy) < cut_mu_dxy) & (events.Muon.mediumId > 0) & (events.Muon.pfRelIso03_all < cut_mu_iso)]

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

        # save info if an extra muon exist (with 0.15 < pfRelIso03_all < 0.4)
        events_tem['nAdditionalMuon'] = ak.num(events_tem.SelMuon[events_tem.SelMuon.pfRelIso03_all > 0.15])

        # events should pass most efficient HLT (for now only 1: IsoMu24)
        events_tem = events_tem[events_tem.HLT.IsoMu24]

        out[f'sumw_HLT'][ds] += ak.sum(events_tem.genWeight)
        out[f'n_ev_HLT'][ds] += len(events_tem)

        events_tem, Sel_Muon = Trigger_Muon_sel(events_tem)

        out[f'sumw_MuTriggerMatching'][ds] += ak.sum(events_tem.genWeight)
        out[f'n_ev_MuTriggerMatching'][ds] += len(events_tem)

        # Select events with only one electron
        cut_one_e = (ak.num(events_tem.SelElectron) == 1) 
        events_tem = events_tem[cut_one_e]
        Sel_Muon = Sel_Muon[cut_one_e]
        Sel_Electron = events_tem.SelElectron[:,0]

        # electron with pt>20
        events_tem = events_tem[Sel_Electron.pt > 20]
        Sel_Muon = Sel_Muon[Sel_Electron.pt > 20]
        Sel_Electron =Sel_Electron[Sel_Electron.pt > 20]

        out[f'sumw_NoAdditionalElectron'][ds] += ak.sum(events_tem.genWeight)
        out[f'n_ev_NoAdditionalElectron'][ds] += len(events_tem)

        # select Tau candidate with dr(muon,Tau)>0.5 and dr(electron,Tau)>0.5  and highest isolation VSJet
        events_tem, Sel_Electron, Sel_Muon, Sel_Tau = FinalTau_sel(events_tem, Sel_Electron, Sel_Muon)

        out[f'sumw_drTauLeptons'][ds] += ak.sum(events_tem.genWeight)
        out[f'n_ev_drTauLeptons'][ds] += len(events_tem)

        bjet_candidate = bjet_candidates(events_tem, Sel_Electron, Sel_Muon, Sel_Tau)

        events_tem['nbjets'] = ak.num(bjet_candidate)
        events_tem['bjets'] = bjet_candidate

        if len(events_tem) == 0:
            return
        
        events_tem['matchingIsoMu24'] = matching_IsoMu24(events_tem, Sel_Muon)

        #computing corrections
        if mode != 'Data':
            pileup_corr = get_pileup_correction(events_tem, 'nominal')* self.norm_factor
            sf_mu = compute_sf_mu(Sel_Muon)
            Trigger_eff_corr_mu = get_trigger_correction_mu(Sel_Muon)
            sf_e = compute_sf_e(Sel_Electron)
            # as we use DeepTau2018v2p5, we don't have corrections yet: only tau energy sf
            sf_tau = compute_sf_tau_e(Sel_Tau)
            events_tem.genWeight = events_tem.genWeight * sf_mu * Trigger_eff_corr_mu * sf_e * sf_tau * pileup_corr

        out[f'sumw_corrections'][ds] += ak.sum(events_tem.genWeight)
        out[f'n_ev_corrections'][ds] += len(events_tem)

        # Save anatuple
        self.save_anatuple_tem(ds, events_tem, Sel_Muon, Sel_Electron, Sel_Tau, self.tag, mode)

        return events_tem
    
    def save_anatuple_tem(self, ds, events, Sel_Muon, Sel_Electron, Sel_Tau, tag, mode):

        exclude_list = ['jetIdxG','genPartIdxG']
        
        save_file, lst = save_anatuple_common(ds, events, tag)
        
        #info specific to the channel
        lst["nAdditionalMuon"] = np.array(events.nAdditionalMuon)
        lst["matchingIsoMu24"] = np.array(events.matchingIsoMu24)

        lst = save_anatuple_lepton(Sel_Muon, lst, exclude_list, 'Muon')
        lst = save_anatuple_lepton(Sel_Electron, lst, exclude_list, 'Electron')
        lst = save_anatuple_tau(events, Sel_Tau, lst, exclude_list, mode, 'Tau')

        save_Event_bjets(save_file, lst, events)
        
        return

    def postprocess(self, accumulator):
        return accumulator