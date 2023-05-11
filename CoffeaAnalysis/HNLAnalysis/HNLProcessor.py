import numpy as np
import awkward as ak

from CoffeaAnalysis.HNLAnalysis.helpers import apply_golden_run, apply_reweight, apply_MET_Filter
from CoffeaAnalysis.HNLAnalysis.correction_helpers import get_pileup_correction, compute_tau_e_corr

class HNLProcessor():
    def __init__(self, stitched_list, tag, xsecs):

        if stitched_list is None or len(stitched_list) == 0:
            raise 'Missing stitched_list in samples_2018.yaml'
        self.stitched_list = stitched_list

        if tag is None or len(tag) == 0:
            raise 'Missing tag'
        self.tag = tag

        if xsecs is None:
            raise 'Missing xsecs'
        self.xsecs = xsecs
    
    # we will receive a NanoEvents instead of a coffea DataFrame
    def init_process(self, out, events):

        ds = events.metadata["dataset"] # dataset name
        print('Processing: ' + ds)

        if self.dataHLT in ds:
            events['genWeight'] = events.run > 0 #setup genWeight for data (1.)

        # initial genWeight received
        out['sumw_init'][ds] += ak.sum(events.genWeight)
        out['n_ev_init'][ds] += len(events)

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
        
        if mode != 'Data':
            #reweights events with lumi x xsec / n_events (with PU correction) + applying stitching weights for DY and WJets samples 
            events = apply_reweight(ds, events, self.stitched_list, self.dataHLT, self.xsecs)
            # pileup correction: compute normalizing factor in order to keep the same number of events before and after correction (before any cut)
            corr = get_pileup_correction(events, 'nominal')
            events['genWeight'] = events.genWeight*corr

        # sumw after lumi x xsec / n_events and PU correction for MC (and "golden" runs selection for data samples)
        out['sumw_reweight'][ds] += ak.sum(events.genWeight)
        out['n_ev_reweight'][ds] += len(events)

        # MET filters
        events = apply_MET_Filter(events)

        # sumw after application of MET filters
        out['sumw_MET_Filter'][ds] += ak.sum(events.genWeight)
        out['n_ev_MET_Filter'][ds] += len(events)

        # Reco event selection: common minimal requirement for leptons
        #tau 
        self.cut_tau_pt = 20. # Tau_pt > cut_tau_pt (general recommendations)
        self.cut_tau_eta = 2.5 #abs(Tau_eta) < cut_tau_eta (general recommendations for DeepTau2p5: 2.3 for DeepTau2p1)
        self.cut_tau_dz = 0.2 #abs(Tau_dz) < cut_tau_dz
        self.cut_tau_idVSmu = 4 # idDeepTau2018v2p5VSmu >= Tight
        self.cut_tau_idVSe = 2 # idDeepTau2018v2p5VSe >= VVLoose
        self.cut_tau_idVSjet = 2 # idDeepTau2018v2p5VSjet >= VVLoose

        #muons
        self.cut_mu_pt = 10. # Muon_pt > cut_mu_pt
        self.cut_mu_eta = 2.4 # abs(Muon_eta) < cut_mu_eta
        self.cut_mu_dz = 0.2 #abs(Muon_dz) < cut_mu_dz
        self.cut_mu_dxy = 0.045 # abs(Muon_dxy) < cut_mu_dxy
        self.cut_mu_iso = 0.4 # Muon_pfRelIso03_all < cut_mu_iso

        #electrons
        self.cut_e_pt = 10. # Electron_pt > cut_e_pt
        self.cut_e_eta = 2.5 # abs(Electron_eta) < cut_e_eta
        self.cut_e_dz = 0.2 #abs(Electron_dz) < cut_e_dz
        self.cut_e_dxy = 0.045 # abs(Electron_dxy) < cut_e_dxy
        self.cut_e_iso = 0.4 # Electron_pfRelIso03_all < cut_e_iso

        return events, out, ds, mode

    def Lepton_selection(self, events, mode):
        #tau
        # cuts in HNLProcessor + remove decay mode 5 and 6 as suggested here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
        events['SelTau'] = events.Tau[(np.abs(events.Tau.eta) < self.cut_tau_eta) & (np.abs(events.Tau.dz) < self.cut_tau_dz) & (events.Tau.idDeepTau2018v2p5VSmu >= self.cut_tau_idVSmu) & (events.Tau.idDeepTau2018v2p5VSe >= self.cut_tau_idVSe) & (events.Tau.idDeepTau2018v2p5VSjet >= self.cut_tau_idVSjet) & (events.Tau.decayMode != 5) & (events.Tau.decayMode != 6)]

        #muons
        # cuts in HNLProcessor + Muon_mediumId
        events['SelMuon'] = events.Muon[(events.Muon.pt > self.cut_mu_pt) & (np.abs(events.Muon.eta) < self.cut_mu_eta) & (np.abs(events.Muon.dz) < self.cut_mu_dz) & (np.abs(events.Muon.dxy) < self.cut_mu_dxy) & (events.Muon.mediumId > 0) & (events.Muon.pfRelIso03_all < self.cut_mu_iso)]

        #electrons
        # cuts in HNLProcessor + mvaNoIso_WP90 > 0 (i.e True)
        events['SelElectron'] = events.Electron[(events.Electron.pt > self.cut_e_pt) & (np.abs(events.Electron.eta) < self.cut_e_eta) & (np.abs(events.Electron.dz) < self.cut_e_dz) & (np.abs(events.Electron.dxy) < self.cut_e_dxy) & (events.Electron.mvaNoIso_WP90 > 0) & (events.Electron.pfRelIso03_all < self.cut_e_iso)]

        #apply energy correction for Tau using DeepTau2017v2p1 values for MC
        if mode != 'Data':
            tau_es   = compute_tau_e_corr(events.SelTau)
            events["SelTau","ptcorr"] = events.SelTau.pt*tau_es
            events["SelTau","masscorr"] = events.SelTau.mass*tau_es
            events['SelTau'] = events.SelTau[events.SelTau.ptcorr > self.cut_tau_pt]
        else:
            events['SelTau'] = events.SelTau[events.SelTau.pt > self.cut_tau_pt]
        
        return events