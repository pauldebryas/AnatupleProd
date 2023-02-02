import numpy as np
import awkward as ak
from coffea import processor, hist

from CoffeaAnalysis.HNLAnalysis.helpers import data_goodrun_lumi, import_stitching_weights, reweight_DY, reweight_WJets
from CoffeaAnalysis.HNLAnalysis.correction_helpers import get_pileup_correction, compute_sf_mu, compute_sf_tau_e, compute_sf_e, get_trigger_correction_mu
from CoffeaAnalysis.HNLAnalysis.helpers import saved_leading_muon, saved_leading_tau, saved_leading_electron, saved_MET, saved_dilepton_mass, saved_drl1l2
from CoffeaAnalysis.HNLAnalysis.helpers import select_lep1_IsoMu24, select_lep2, select_lep3, bjet_veto, charge_veto, met_veto, z_veto_tll

class HNLAnalysis_tem_OS(processor.ProcessorABC):
    def __init__(self, region, stitched_list):
        ds_axis = hist.Cat("ds", "Primary dataset")
        acc_dict = {var: hist.Hist("Counts", ds_axis, axis) for var, axis in self.get_var_axis_pairs()}
        acc_dict[f'n_ev_all'] = processor.defaultdict_accumulator(int)
        acc_dict[f'sumw_all'] = processor.defaultdict_accumulator(float)
        self.selections = self.get_selections()
        for selection in self.selections:
            acc_dict[f'n_ev_{selection}'] = processor.defaultdict_accumulator(int)
            acc_dict[f'sumw_{selection}'] = processor.defaultdict_accumulator(float)

        acc_dict['sel_array'] = processor.column_accumulator(np.ndarray((0, 12)))
        self._accumulator = processor.dict_accumulator(acc_dict)
        if region not in ['A','B','C','D']:
            raise 'Value error for region argument: can be A, B, C or D'
        self.region = region
        if stitched_list is None or len(stitched_list) == 0:
            raise 'Missing stitched_list in samples_2018.yaml'
        self.stitched_list = stitched_list
    
    @property
    def accumulator(self):
        return self._accumulator

    @staticmethod
    def get_selections():
        return [
            '3leptons',
            'hlt',
            'l1sel',
            'l2sel',
            'l3sel',
            'corrections',
            'bjetveto',
            'chargeveto',
            'metselection',
            'zveto'
        ]
    
    @staticmethod
    def get_var_axis_pairs():

        pt_axis = hist.Bin("pt", r"$p_{T}$ [GeV]", 300, 0., 1500)
        eta_axis = hist.Bin('eta', r'$\eta$', 30,  -3.1415927, 3.1415927)
        phi_axis = hist.Bin('phi', r'$\phi$', 30, -3.1415927, 3.1415927)
        mass_axis = hist.Bin("mass", r"$m_{\ell\ell}$ [GeV]", 300, 0., 1500.)
        dr_axis = hist.Bin("dr", r"$\Delta R$", 30, 0., 5.)
        mc_axis = hist.Bin("mc", r"Combined mass [GeV]", 300, 0., 1500.)
        met_axis = hist.Bin("met", r"PF MET [GeV]", 30, 0., 300.)
        mt_axis = hist.Bin("mt", r"Transverse mass [GeV]", 300, 0., 1500.)

        v_a_pairs = [
            ('pt_tau1', pt_axis),
            ('eta_tau1', eta_axis),
            ('phi_tau1', phi_axis),
            ('mass_tau1', mass_axis),
            ('pt_tau2', pt_axis),
            ('eta_tau2', eta_axis),
            ('phi_tau2', phi_axis),
            ('mass_tau2', mass_axis),
            ('pt_tau3', pt_axis),
            ('eta_tau3', eta_axis),
            ('phi_tau3', phi_axis),
            ('mass_tau3', mass_axis),
            ('pt_mu1', pt_axis),
            ('eta_mu1', eta_axis),
            ('phi_mu1', phi_axis),
            ('mass_mu1', mass_axis),
            ('pt_mu2', pt_axis),
            ('eta_mu2', eta_axis),
            ('phi_mu2', phi_axis),
            ('mass_mu2', mass_axis),
            ('pt_e1', pt_axis),
            ('eta_e1', eta_axis),
            ('phi_e1', phi_axis),
            ('mass_e1', mass_axis),
            ('pt_e2', pt_axis),
            ('eta_e2', eta_axis),
            ('phi_e2', phi_axis),
            ('mass_e2', mass_axis),

            ('dr_l1l2', dr_axis),
            ('comb_mass_l1l2', mc_axis),
            ('comb_mass_taul1', mc_axis),
            ('met', met_axis),
            ('pt_sum_l1l2l3', pt_axis),
            ('pt_sum_l1l2MET', pt_axis),
            ('mT_tautau', mt_axis),
            ('mT_l1MET', mt_axis),
        ]

        return v_a_pairs
    
    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        ds = events.metadata["dataset"] # dataset name

        # for tte channel we only consider Tau dataset
        if 'Tau_2018' in ds:
            return out
        if 'EGamma_2018' in ds:
            return out

        #stitching
        if ds in self.stitched_list['DY_samples']:
            stitching_weights_DY = import_stitching_weights('DYtoLL')
            events = reweight_DY(events, stitching_weights_DY)
            
        if ds in self.stitched_list['WJets_samples']:
            stitching_weights_WJets = import_stitching_weights('WJetsToLNu')
            events = reweight_WJets(events, stitching_weights_WJets)

        # defining the mode
        mode ='MCbackground' # default mode

        if 'SingleMuon_2018' in ds:
            # only keep the good runs
            goodrun_lumi = data_goodrun_lumi(ds)
            goodrun = goodrun_lumi[:,0]
            events = events[np.isin(events.run, goodrun)]
            events['genWeight'] = events.run > 0
            ds = ds[0:-1]
            mode ='Data'
        
        if 'HNL' in ds:
            mode ='signal'
        
        out['sumw_all'][ds] += ak.sum(events.genWeight)
        out['n_ev_all'][ds] += len(events)

        # pileup correction: compute normalizing factor in order to keep the same number of events before and after correction (before any cut)
        if mode != 'Data':
            corr = get_pileup_correction(events, 'nominal')
            self.norm_factor = ak.sum(events.genWeight)/ak.sum(events.genWeight*corr)

        # MET filter folowing https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#2018_2017_data_and_MC_UL
        events = events[events.Flag.goodVertices & 
                        events.Flag.globalSuperTightHalo2016Filter & 
                        events.Flag.HBHENoiseFilter & 
                        events.Flag.HBHENoiseIsoFilter & 
                        events.Flag.EcalDeadCellTriggerPrimitiveFilter & 
                        events.Flag.BadPFMuonFilter & 
                        events.Flag.BadPFMuonDzFilter & 
                        #events.Flag.hfNoisyHitsFilter & 
                        events.Flag.eeBadScFilter & 
                        events.Flag.ecalBadCalibFilter]

        # Reco event selection: common minimal requirement for leptons
        #tau
        cut_tau_pt = 20. # Tau_pt > cut_tau_pt
        cut_tau_eta = 2.5 #abs(Tau_eta) < cut_tau_eta
        cut_tau_dz = 0.2 #abs(Tau_dz) < cut_tau_dz
        cut_tau_idVSmu = 4 # idDeepTau2018v2p5VSmu >= Tight
        cut_tau_idVSe = 3 # idDeepTau2018v2p5VSe >= VLoose
        cut_tau_idVSjet = 5 # idDeepTau2018v2p5VSjet >= Medium
    
        #electrons
        cut_e_pt = 25. # Electron_pt > cut_e_pt
        cut_e_eta = 2.5 # abs(Electron_eta) < cut_e_eta
        cut_e_dz = 0.2 #abs(Electron_dz) < cut_e_dz
        cut_e_dxy = 0.045 # abs(Electron_dxy) < cut_e_dxy
        cut_e_id = 0 # Electron_mvaIso_WP90 > cut_e_id (i.e True)
        
        #muons
        cut_mu_pt = 25. # Muon_pt > cut_mu_pt
        cut_mu_eta = 2.4 # abs(Muon_eta) < cut_mu_eta
        cut_mu_dz = 0.2 #abs(Muon_dz) < cut_mu_dz
        cut_mu_dxy = 0.045 # abs(Muon_dxy) < cut_mu_dxy
        cut_mu_id = 0 #(Muon_mediumId > cut_mu_id || Muon_tightId > cut_mu_id)
        cut_mu_iso = 0.15 # Muon_pfRelIso03_all < cut_mu_iso

        if self.region == 'A':
            #tau
            cut_tau_idVSjet_low = 3 # VLoose <= Tau_idDeepTau2017v2p1VSjet 
            cut_tau_idVSjet_high = 5 # Tau_idDeepTau2017v2p1VSjet < Medium
            # + remove decay mode 5 and 6 as suggested here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
            events['SelTau'] = events.Tau[(events.Tau.pt > cut_tau_pt) & (np.abs(events.Tau.eta) < cut_tau_eta) & (np.abs(events.Tau.dz) < cut_tau_dz) & (events.Tau.idDeepTau2018v2p5VSmu >= cut_tau_idVSmu) & (events.Tau.idDeepTau2018v2p5VSe >= cut_tau_idVSe) & (events.Tau.idDeepTau2018v2p5VSjet < cut_tau_idVSjet_high) & (events.Tau.idDeepTau2018v2p5VSjet >= cut_tau_idVSjet_low) & (events.Tau.decayMode != 5) & (events.Tau.decayMode != 6)]

            #electrons
            cut_e_id = 0 # Electron_mvaIso_WP90 > cut_e_id (i.e True)
            events['SelElectron'] = events.Electron[(events.Electron.pt > cut_e_pt) & (np.abs(events.Electron.eta) < cut_e_eta) & (np.abs(events.Electron.dz) < cut_e_dz) & (np.abs(events.Electron.dxy) < cut_e_dxy) & (events.Electron.mvaIso_WP90 > cut_e_id)]

            #muons
            cut_mu_iso_high = 0.4 # cut_mu_iso_low < Muon_pfRelIso03_all
            cut_mu_iso_low = 0.2 # Muon_pfRelIso03_all < cut_mu_iso_high
            events['SelMuon'] = events.Muon[(events.Muon.pt > cut_mu_pt) & (np.abs(events.Muon.eta) < cut_mu_eta) & (np.abs(events.Muon.dz) < cut_mu_dz) & (np.abs(events.Muon.dxy) < cut_mu_dxy) & ((events.Muon.mediumId > cut_mu_id) | (events.Muon.tightId > cut_mu_id)) & (events.Muon.pfRelIso03_all < cut_mu_iso_high) & (events.Muon.pfRelIso03_all > cut_mu_iso_low)]
            
        if self.region == 'B':
            #tau
            cut_tau_idVSjet = 5 # idDeepTau2018v2p5VSjet >= Medium
            # + remove decay mode 5 and 6 as suggested here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
            events['SelTau'] = events.Tau[(events.Tau.pt > cut_tau_pt) & (np.abs(events.Tau.eta) < cut_tau_eta) & (np.abs(events.Tau.dz) < cut_tau_dz) & (events.Tau.idDeepTau2018v2p5VSmu >= cut_tau_idVSmu) & (events.Tau.idDeepTau2018v2p5VSe >= cut_tau_idVSe) & (events.Tau.idDeepTau2018v2p5VSjet >= cut_tau_idVSjet) & (events.Tau.decayMode != 5) & (events.Tau.decayMode != 6)]

            #electrons
            cut_e_id = 0 # Electron_mvaIso_WP90 > cut_e_id (i.e True)
            events['SelElectron'] = events.Electron[(events.Electron.pt > cut_e_pt) & (np.abs(events.Electron.eta) < cut_e_eta) & (np.abs(events.Electron.dz) < cut_e_dz) & (np.abs(events.Electron.dxy) < cut_e_dxy) & (events.Electron.mvaIso_WP90 > cut_e_id)]

            #muons
            cut_mu_iso_high = 0.4 # cut_mu_iso_low < Muon_pfRelIso03_all
            cut_mu_iso_low = 0.2 # Muon_pfRelIso03_all < cut_mu_iso_high
            events['SelMuon'] = events.Muon[(events.Muon.pt > cut_mu_pt) & (np.abs(events.Muon.eta) < cut_mu_eta) & (np.abs(events.Muon.dz) < cut_mu_dz) & (np.abs(events.Muon.dxy) < cut_mu_dxy) & ((events.Muon.mediumId > cut_mu_id) | (events.Muon.tightId > cut_mu_id)) & (events.Muon.pfRelIso03_all < cut_mu_iso_high) & (events.Muon.pfRelIso03_all > cut_mu_iso_low)]

        if self.region == 'C':
            #tau
            cut_tau_idVSjet_low = 3 # VLoose <= Tau_idDeepTau2017v2p1VSjet 
            cut_tau_idVSjet_high = 5 # Tau_idDeepTau2017v2p1VSjet < Medium
            # + remove decay mode 5 and 6 as suggested here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
            events['SelTau'] = events.Tau[(events.Tau.pt > cut_tau_pt) & (np.abs(events.Tau.eta) < cut_tau_eta) & (np.abs(events.Tau.dz) < cut_tau_dz) & (events.Tau.idDeepTau2018v2p5VSmu >= cut_tau_idVSmu) & (events.Tau.idDeepTau2018v2p5VSe >= cut_tau_idVSe) & (events.Tau.idDeepTau2018v2p5VSjet < cut_tau_idVSjet_high) & (events.Tau.idDeepTau2018v2p5VSjet >= cut_tau_idVSjet_low) & (events.Tau.decayMode != 5) & (events.Tau.decayMode != 6)]

            #electrons
            cut_e_id = 0 # Electron_mvaIso_WP90 > cut_e_id (i.e True)
            events['SelElectron'] = events.Electron[(events.Electron.pt > cut_e_pt) & (np.abs(events.Electron.eta) < cut_e_eta) & (np.abs(events.Electron.dz) < cut_e_dz) & (np.abs(events.Electron.dxy) < cut_e_dxy) & (events.Electron.mvaIso_WP90 > cut_e_id)]

            #muons
            cut_mu_iso = 0.15 # Muon_pfRelIso03_all < cut_mu_iso
            events['SelMuon'] = events.Muon[(events.Muon.pt > cut_mu_pt) & (np.abs(events.Muon.eta) < cut_mu_eta) & (np.abs(events.Muon.dz) < cut_mu_dz) & (np.abs(events.Muon.dxy) < cut_mu_dxy) & ((events.Muon.mediumId > cut_mu_id) | (events.Muon.tightId > cut_mu_id)) & (events.Muon.pfRelIso03_all < cut_mu_iso)]

        if self.region == 'D':
            #tau
            cut_tau_idVSjet = 5 # idDeepTau2018v2p5VSjet >= Medium
            # + remove decay mode 5 and 6 as suggested here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
            events['SelTau'] = events.Tau[(events.Tau.pt > cut_tau_pt) & (np.abs(events.Tau.eta) < cut_tau_eta) & (np.abs(events.Tau.dz) < cut_tau_dz) & (events.Tau.idDeepTau2018v2p5VSmu >= cut_tau_idVSmu) & (events.Tau.idDeepTau2018v2p5VSe >= cut_tau_idVSe) & (events.Tau.idDeepTau2018v2p5VSjet >= cut_tau_idVSjet) & (events.Tau.decayMode != 5) & (events.Tau.decayMode != 6)]

            #electrons
            cut_e_id = 0 # Electron_mvaIso_WP90 > cut_e_id (i.e True)
            events['SelElectron'] = events.Electron[(events.Electron.pt > cut_e_pt) & (np.abs(events.Electron.eta) < cut_e_eta) & (np.abs(events.Electron.dz) < cut_e_dz) & (np.abs(events.Electron.dxy) < cut_e_dxy) & (events.Electron.mvaIso_WP90 > cut_e_id)]

            #muons
            cut_mu_iso = 0.15 # Muon_pfRelIso03_all < cut_mu_iso
            events['SelMuon'] = events.Muon[(events.Muon.pt > cut_mu_pt) & (np.abs(events.Muon.eta) < cut_mu_eta) & (np.abs(events.Muon.dz) < cut_mu_dz) & (np.abs(events.Muon.dxy) < cut_mu_dxy) & ((events.Muon.mediumId > cut_mu_id) | (events.Muon.tightId > cut_mu_id)) & (events.Muon.pfRelIso03_all < cut_mu_iso)]

        self.analyse_tem_OS(events, out, ds, mode)

        return out

    def analyse_tem_OS(self, events, out, ds, mode):

        # select tem_OS events: require 1 reco mu, 1 reco tau and 1 reco e + e and mu have the OS
        events_tem_OS = events[(ak.num(events.SelElectron) == 1) & (ak.num(events.SelMuon) == 1) & (ak.num(events.SelTau) == 1)] 
        events_tem_OS = events_tem_OS[ak.flatten(events_tem_OS.SelElectron.charge != events_tem_OS.SelMuon.charge)]

        out[f'sumw_3leptons'][ds] += ak.sum(events_tem_OS.genWeight)
        out[f'n_ev_3leptons'][ds] += len(events_tem_OS)

        # events should pass most efficient HLT (for now only 1: IsoMu24)
        events_tem_OS = events_tem_OS[events_tem_OS.HLT.IsoMu24]

        out[f'sumw_hlt'][ds] += ak.sum(events_tem_OS.genWeight)
        out[f'n_ev_hlt'][ds] += len(events_tem_OS)

        if len(events_tem_OS) == 0:
            return

        # select reco muon that match HLT IsoMu24
        Sel_Muon = select_lep1_IsoMu24(events_tem_OS, min_dr_cut=0.2)

        #removing non matching events
        events_tem_OS = events_tem_OS[(ak.num(Sel_Muon) == 1)]
        Sel_Muon = Sel_Muon[(ak.num(Sel_Muon) == 1)]

        out[f'sumw_l1sel'][ds] += ak.sum(events_tem_OS.genWeight)
        out[f'n_ev_l1sel'][ds] += len(events_tem_OS)

        # select reco Electron with dr(e,mu)>0.5 (in case there is more than 1 selected we choose the one with higher pt )
        Sel_Electron = select_lep2(events_tem_OS, Sel_Muon, type='electron', delta_r_cut = 0.5)

        #removing non matching events
        Sel_Muon = Sel_Muon[(ak.num(Sel_Electron) == 1)]
        events_tem_OS = events_tem_OS[(ak.num(Sel_Electron) == 1)]
        Sel_Electron = Sel_Electron[(ak.num(Sel_Electron) == 1)]

        out[f'sumw_l2sel'][ds] += ak.sum(events_tem_OS.genWeight)
        out[f'n_ev_l2sel'][ds] += len(events_tem_OS)

        # select reco Tau with dr(tau,mu)>0.5 and dr(tau,electron)>0.5 (in case there is more than 1 selected we choose the one with higher pt )
        Sel_Tau = select_lep3(events_tem_OS, Sel_Muon, Sel_Electron, type='tau', delta_r_cut = 0.5)

        #removing non matching events
        Sel_Muon = Sel_Muon[(ak.num(Sel_Tau) == 1)]
        Sel_Electron = Sel_Electron[(ak.num(Sel_Tau) == 1)]
        events_tem_OS = events_tem_OS[(ak.num(Sel_Tau) == 1)]
        Sel_Tau = Sel_Tau[(ak.num(Sel_Tau) == 1)]
        
        out[f'sumw_l3sel'][ds] += ak.sum(events_tem_OS.genWeight)
        out[f'n_ev_l3sel'][ds] += len(events_tem_OS)
        '''
        events_tem_OS, Sel_Tau, Sel_Electron, Sel_Muon = bjet_veto(events_tem_OS, Sel_Tau, Sel_Electron, Sel_Muon)
        out[f'sumw_bjetveto'][ds] += ak.sum(events_tem_OS.genWeight)
        out[f'n_ev_bjetveto'][ds] += len(events_tem_OS)

        events_tem_OS, Sel_Tau, Sel_Electron, Sel_Muon = charge_veto(events_tem_OS, Sel_Tau, Sel_Electron, Sel_Muon)
        out[f'sumw_chargeveto'][ds] += ak.sum(events_tem_OS.genWeight)
        out[f'n_ev_chargeveto'][ds] += len(events_tem_OS)

        events_tem_OS, Sel_Tau, Sel_Electron, Sel_Muon = met_veto(events_tem_OS, Sel_Tau, Sel_Electron, Sel_Muon)
        out[f'sumw_metselection'][ds] += ak.sum(events_tem_OS.genWeight)
        out[f'n_ev_metselection'][ds] += len(events_tem_OS)

        events_tem_OS, Sel_Tau, Sel_Electron, Sel_Muon = z_veto_tll(events_tem_OS, Sel_Tau, Sel_Electron, Sel_Muon)
        out[f'sumw_zveto'][ds] += ak.sum(events_tem_OS.genWeight)
        out[f'n_ev_zveto'][ds] += len(events_tem_OS)
       '''
       #computing corrections
        if mode != 'Data':
            pileup_corr = get_pileup_correction(events_tem_OS, 'nominal')* self.norm_factor
            sf_mu = compute_sf_mu(Sel_Muon)
            Trigger_eff_corr_mu = get_trigger_correction_mu(Sel_Muon)
            sf_e = compute_sf_e(Sel_Electron)
            # as we use DeepTau2018v2p5, we don't have corrections yet
            sf_tau = compute_sf_tau_e(Sel_Tau)
            events_tem_OS.genWeight = events_tem_OS.genWeight * sf_mu * Trigger_eff_corr_mu * sf_e * sf_tau * pileup_corr

        out[f'sumw_corrections'][ds] += ak.sum(events_tem_OS.genWeight)
        out[f'n_ev_corrections'][ds] += len(events_tem_OS)

        # Save histograms
        saved_leading_muon(events_tem_OS, Sel_Muon, out, ds)
        saved_leading_tau(events_tem_OS, Sel_Tau, out, ds)
        saved_leading_electron(events_tem_OS, Sel_Electron, out, ds)
        saved_MET(events_tem_OS, out, ds)
        saved_dilepton_mass(events_tem_OS, Sel_Electron, Sel_Muon, out, ds)
        saved_drl1l2(events_tem_OS, Sel_Muon, Sel_Electron, out, ds)

        return events_tem_OS

    def postprocess(self, accumulator):
        return accumulator
