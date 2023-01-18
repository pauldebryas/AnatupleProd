import numpy as np
import awkward as ak
from coffea import processor, hist
from CoffeaAnalysis.HNLAnalysis.helpers import delta_r, data_goodrun_lumi, import_stitching_weights
from CoffeaAnalysis.HNLAnalysis.correction_helpers import get_pileup_correction, compute_sf_mu, compute_sf_tau_e, get_trigger_correction_mu

class HNLAnalysis_tmm(processor.ProcessorABC):
    def __init__(self, region):
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
        DY_samples =['DYJetsToLL_M-50',
                    'DYJetsToLL_0J',
                    'DYJetsToLL_1J',
                    'DYJetsToLL_2J',
                    'DYJetsToLL_LHEFilterPtZ-0To50',
                    'DYJetsToLL_LHEFilterPtZ-50To100',
                    'DYJetsToLL_LHEFilterPtZ-100To250',
                    'DYJetsToLL_LHEFilterPtZ-250To400',
                    'DYJetsToLL_LHEFilterPtZ-400To650',
                    'DYJetsToLL_LHEFilterPtZ-650ToInf']

        WJets_samples = ['WJetsToLNu',
                        'W1JetsToLNu',
                        'W2JetsToLNu',
                        'W3JetsToLNu',
                        'W4JetsToLNu',
                        'WJetsToLNu_HT-70To100',
                        'WJetsToLNu_HT-100To200',
                        'WJetsToLNu_HT-200To400',
                        'WJetsToLNu_HT-400To600', 
                        'WJetsToLNu_HT-600To800', 
                        'WJetsToLNu_HT-800To1200', 
                        'WJetsToLNu_HT-1200To2500', 
                        'WJetsToLNu_HT-2500ToInf']
        
        if ds in DY_samples:
            stitching_weights_DY = import_stitching_weights('DYtoLL')

            np.asarray(events.genWeight)[events.genWeight < 0] = -1.
            np.asarray(events.genWeight)[events.genWeight > 0] = 1.
            np.asarray(events.genWeight)[events.genWeight == 0] = 0.

            for NJets in np.arange(len(list(stitching_weights_DY))):
                np.asarray(events.genWeight)[(events.LHE.Vpt == 0) & (events.LHE.NpNLO == int(NJets))] = events.genWeight[(events.LHE.Vpt == 0) & (events.LHE.NpNLO == int(NJets))]*stitching_weights_DY['NJets='+str(NJets)]['PtZ=0']
                np.asarray(events.genWeight)[(events.LHE.Vpt > 0) & (events.LHE.Vpt < 50) & (events.LHE.NpNLO == int(NJets))] = events.genWeight[(events.LHE.Vpt > 0) & (events.LHE.Vpt < 50) & (events.LHE.NpNLO == int(NJets))]*stitching_weights_DY['NJets='+str(NJets)]['PtZ=0to50']
                np.asarray(events.genWeight)[(events.LHE.Vpt >= 50) & (events.LHE.Vpt < 100) & (events.LHE.NpNLO == int(NJets))] = events.genWeight[(events.LHE.Vpt >= 50) & (events.LHE.Vpt < 100) & (events.LHE.NpNLO == int(NJets))]*stitching_weights_DY['NJets='+str(NJets)]['PtZ=50to100']
                np.asarray(events.genWeight)[(events.LHE.Vpt >= 100) & (events.LHE.Vpt < 250) & (events.LHE.NpNLO == int(NJets))] = events.genWeight[(events.LHE.Vpt >= 100) & (events.LHE.Vpt < 250) & (events.LHE.NpNLO == int(NJets))]*stitching_weights_DY['NJets='+str(NJets)]['PtZ=100to250']
                np.asarray(events.genWeight)[(events.LHE.Vpt >= 250) & (events.LHE.Vpt < 400) & (events.LHE.NpNLO == int(NJets))] = events.genWeight[(events.LHE.Vpt >= 250) & (events.LHE.Vpt < 400) & (events.LHE.NpNLO == int(NJets))]*stitching_weights_DY['NJets='+str(NJets)]['PtZ=250to400']
                np.asarray(events.genWeight)[(events.LHE.Vpt >= 400) & (events.LHE.Vpt < 650) & (events.LHE.NpNLO == int(NJets))] = events.genWeight[(events.LHE.Vpt >= 400) & (events.LHE.Vpt < 650) & (events.LHE.NpNLO == int(NJets))]*stitching_weights_DY['NJets='+str(NJets)]['PtZ=400to650']
                np.asarray(events.genWeight)[(events.LHE.Vpt >= 650) & (events.LHE.NpNLO == int(NJets))] = events.genWeight[(events.LHE.Vpt >= 650) & (events.LHE.NpNLO == int(NJets))]*stitching_weights_DY['NJets='+str(NJets)]['PtZ=650toInf']

        if ds in WJets_samples:
            stitching_weights_WJets = import_stitching_weights('WJetsToLNu')

            np.asarray(events.genWeight)[events.genWeight < 0] = -1.
            np.asarray(events.genWeight)[events.genWeight > 0] = 1.
            np.asarray(events.genWeight)[events.genWeight == 0] = 0.

            for NJets in np.arange(len(list(stitching_weights_WJets))):
                np.asarray(events.genWeight)[(events.LHE.HT == 0) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT == 0) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=0']
                np.asarray(events.genWeight)[(events.LHE.HT > 0) & (events.LHE.HT < 70) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT > 0) & (events.LHE.HT < 70) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=0to70']
                np.asarray(events.genWeight)[(events.LHE.HT >= 70) & (events.LHE.HT < 100) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT >= 70) & (events.LHE.HT < 100) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=70to100']
                np.asarray(events.genWeight)[(events.LHE.HT >= 100) & (events.LHE.HT < 200) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT >= 100) & (events.LHE.HT < 200) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=100to200']
                np.asarray(events.genWeight)[(events.LHE.HT >= 200) & (events.LHE.HT < 400) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT >= 200) & (events.LHE.HT < 400) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=200to400']
                np.asarray(events.genWeight)[(events.LHE.HT >= 400) & (events.LHE.HT < 600) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT >= 400) & (events.LHE.HT < 600) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=400to600']
                np.asarray(events.genWeight)[(events.LHE.HT >= 600) & (events.LHE.HT < 800) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT >= 600) & (events.LHE.HT < 800) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=600to800']
                np.asarray(events.genWeight)[(events.LHE.HT >= 800) & (events.LHE.HT < 1200) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT >= 800) & (events.LHE.HT < 1200) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=800to1200']
                np.asarray(events.genWeight)[(events.LHE.HT >= 1200) & (events.LHE.HT < 2500) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT >= 1200) & (events.LHE.HT < 2500) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=1200to2500']
                np.asarray(events.genWeight)[(events.LHE.HT >= 2500) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT >= 2500) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=2500toInf']
            
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

        if self.region == 'A':
            # Reco event selection: minimal requirement for leptons
            #tau
            cut_tau_pt = 20. # Tau_pt > cut_tau_pt
            cut_tau_eta = 2.5 #abs(Tau_eta) < cut_tau_eta
            cut_tau_dz = 0.2 #abs(Tau_dz) < cut_tau_dz
            cut_tau_idVSmu = 4 # idDeepTau2018v2p5VSmu >= Tight
            cut_tau_idVSe = 3 # idDeepTau2018v2p5VSe >= VLoose
            cut_tau_idVSjet_low = 3 # VLoose <= Tau_idDeepTau2017v2p1VSjet 
            cut_tau_idVSjet_high = 5 # Tau_idDeepTau2017v2p1VSjet < Medium
            # + remove decay mode 5 and 6 as suggested here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
            events['SelTau'] = events.Tau[(events.Tau.pt > cut_tau_pt) & (np.abs(events.Tau.eta) < cut_tau_eta) & (np.abs(events.Tau.dz) < cut_tau_dz) & (events.Tau.idDeepTau2018v2p5VSmu >= cut_tau_idVSmu) & (events.Tau.idDeepTau2018v2p5VSe >= cut_tau_idVSe) & (events.Tau.idDeepTau2018v2p5VSjet < cut_tau_idVSjet_high) & (events.Tau.idDeepTau2018v2p5VSjet >= cut_tau_idVSjet_low) & (events.Tau.decayMode != 5) & (events.Tau.decayMode != 6)]

            #electrons
            cut_e_pt = 25. # Electron_pt > cut_e_pt
            cut_e_eta = 2.5 # abs(Electron_eta) < cut_e_eta
            cut_e_dz = 0.2 #abs(Electron_dz) < cut_e_dz
            cut_e_dxy = 0.045 # abs(Electron_dxy) < cut_e_dxy
            cut_e_id = 0 # Electron_mvaIso_WP90 > cut_e_id (i.e True)
            events['SelElectron'] = events.Electron[(events.Electron.pt > cut_e_pt) & (np.abs(events.Electron.eta) < cut_e_eta) & (np.abs(events.Electron.dz) < cut_e_dz) & (np.abs(events.Electron.dxy) < cut_e_dxy) & (events.Electron.mvaIso_WP90 > cut_e_id)]

            #muons
            cut_mu_pt = 25. # Muon_pt > cut_mu_pt
            cut_mu_eta = 2.4 # abs(Muon_eta) < cut_mu_eta
            cut_mu_dz = 0.2 #abs(Muon_dz) < cut_mu_dz
            cut_mu_dxy = 0.045 # abs(Muon_dxy) < cut_mu_dxy
            cut_mu_id = 0 #(Muon_mediumId > cut_mu_id || Muon_tightId > cut_mu_id)
            cut_mu_iso_high = 0.4 # cut_mu_iso_low < Muon_pfRelIso03_all
            cut_mu_iso_low = 0.2 # Muon_pfRelIso03_all < cut_mu_iso_high
            events['SelMuon'] = events.Muon[(events.Muon.pt > cut_mu_pt) & (np.abs(events.Muon.eta) < cut_mu_eta) & (np.abs(events.Muon.dz) < cut_mu_dz) & (np.abs(events.Muon.dxy) < cut_mu_dxy) & ((events.Muon.mediumId > cut_mu_id) | (events.Muon.tightId > cut_mu_id)) & (events.Muon.pfRelIso03_all < cut_mu_iso_high) & (events.Muon.pfRelIso03_all > cut_mu_iso_low)]

        if self.region == 'B':
            # Reco event selection: minimal requirement for leptons
            #tau
            cut_tau_pt = 20. # Tau_pt > cut_tau_pt
            cut_tau_eta = 2.5 #abs(Tau_eta) < cut_tau_eta
            cut_tau_dz = 0.2 #abs(Tau_dz) < cut_tau_dz
            cut_tau_idVSmu = 4 # idDeepTau2018v2p5VSmu >= Tight
            cut_tau_idVSe = 3 # idDeepTau2018v2p5VSe >= VLoose
            cut_tau_idVSjet = 5 # idDeepTau2018v2p5VSjet >= Medium
            # + remove decay mode 5 and 6 as suggested here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
            events['SelTau'] = events.Tau[(events.Tau.pt > cut_tau_pt) & (np.abs(events.Tau.eta) < cut_tau_eta) & (np.abs(events.Tau.dz) < cut_tau_dz) & (events.Tau.idDeepTau2018v2p5VSmu >= cut_tau_idVSmu) & (events.Tau.idDeepTau2018v2p5VSe >= cut_tau_idVSe) & (events.Tau.idDeepTau2018v2p5VSjet >= cut_tau_idVSjet) & (events.Tau.decayMode != 5) & (events.Tau.decayMode != 6)]

            #electrons
            cut_e_pt = 25. # Electron_pt > cut_e_pt
            cut_e_eta = 2.5 # abs(Electron_eta) < cut_e_eta
            cut_e_dz = 0.2 #abs(Electron_dz) < cut_e_dz
            cut_e_dxy = 0.045 # abs(Electron_dxy) < cut_e_dxy
            cut_e_id = 0 # Electron_mvaIso_WP90 > cut_e_id (i.e True)
            events['SelElectron'] = events.Electron[(events.Electron.pt > cut_e_pt) & (np.abs(events.Electron.eta) < cut_e_eta) & (np.abs(events.Electron.dz) < cut_e_dz) & (np.abs(events.Electron.dxy) < cut_e_dxy) & (events.Electron.mvaIso_WP90 > cut_e_id)]

            #muons
            cut_mu_pt = 25. # Muon_pt > cut_mu_pt
            cut_mu_eta = 2.4 # abs(Muon_eta) < cut_mu_eta
            cut_mu_dz = 0.2 #abs(Muon_dz) < cut_mu_dz
            cut_mu_dxy = 0.045 # abs(Muon_dxy) < cut_mu_dxy
            cut_mu_id = 0 #(Muon_mediumId > cut_mu_id || Muon_tightId > cut_mu_id)
            cut_mu_iso_high = 0.4 # cut_mu_iso_low < Muon_pfRelIso03_all
            cut_mu_iso_low = 0.2 # Muon_pfRelIso03_all < cut_mu_iso_high
            events['SelMuon'] = events.Muon[(events.Muon.pt > cut_mu_pt) & (np.abs(events.Muon.eta) < cut_mu_eta) & (np.abs(events.Muon.dz) < cut_mu_dz) & (np.abs(events.Muon.dxy) < cut_mu_dxy) & ((events.Muon.mediumId > cut_mu_id) | (events.Muon.tightId > cut_mu_id)) & (events.Muon.pfRelIso03_all < cut_mu_iso_high) & (events.Muon.pfRelIso03_all > cut_mu_iso_low)]

        if self.region == 'C':
            # Reco event selection: minimal requirement for leptons
            #tau
            cut_tau_pt = 20. # Tau_pt > cut_tau_pt
            cut_tau_eta = 2.5 #abs(Tau_eta) < cut_tau_eta
            cut_tau_dz = 0.2 #abs(Tau_dz) < cut_tau_dz
            cut_tau_idVSmu = 4 # idDeepTau2018v2p5VSmu >= Tight
            cut_tau_idVSe = 3 # idDeepTau2018v2p5VSe >= VLoose
            cut_tau_idVSjet_low = 3 # VLoose <= Tau_idDeepTau2017v2p1VSjet 
            cut_tau_idVSjet_high = 5 # Tau_idDeepTau2017v2p1VSjet < Medium
            # + remove decay mode 5 and 6 as suggested here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
            events['SelTau'] = events.Tau[(events.Tau.pt > cut_tau_pt) & (np.abs(events.Tau.eta) < cut_tau_eta) & (np.abs(events.Tau.dz) < cut_tau_dz) & (events.Tau.idDeepTau2018v2p5VSmu >= cut_tau_idVSmu) & (events.Tau.idDeepTau2018v2p5VSe >= cut_tau_idVSe) & (events.Tau.idDeepTau2018v2p5VSjet < cut_tau_idVSjet_high) & (events.Tau.idDeepTau2018v2p5VSjet >= cut_tau_idVSjet_low) & (events.Tau.decayMode != 5) & (events.Tau.decayMode != 6)]

            #electrons
            cut_e_pt = 25. # Electron_pt > cut_e_pt
            cut_e_eta = 2.5 # abs(Electron_eta) < cut_e_eta
            cut_e_dz = 0.2 #abs(Electron_dz) < cut_e_dz
            cut_e_dxy = 0.045 # abs(Electron_dxy) < cut_e_dxy
            cut_e_id = 0 # Electron_mvaIso_WP90 > cut_e_id (i.e True)
            events['SelElectron'] = events.Electron[(events.Electron.pt > cut_e_pt) & (np.abs(events.Electron.eta) < cut_e_eta) & (np.abs(events.Electron.dz) < cut_e_dz) & (np.abs(events.Electron.dxy) < cut_e_dxy) & (events.Electron.mvaIso_WP90 > cut_e_id)]

            #muons
            cut_mu_pt = 25. # Muon_pt > cut_mu_pt
            cut_mu_eta = 2.4 # abs(Muon_eta) < cut_mu_eta
            cut_mu_dz = 0.2 #abs(Muon_dz) < cut_mu_dz
            cut_mu_dxy = 0.045 # abs(Muon_dxy) < cut_mu_dxy
            cut_mu_id = 0 #(Muon_mediumId > cut_mu_id || Muon_tightId > cut_mu_id)
            cut_mu_iso = 0.15 # Muon_pfRelIso03_all < cut_mu_iso
            events['SelMuon'] = events.Muon[(events.Muon.pt > cut_mu_pt) & (np.abs(events.Muon.eta) < cut_mu_eta) & (np.abs(events.Muon.dz) < cut_mu_dz) & (np.abs(events.Muon.dxy) < cut_mu_dxy) & ((events.Muon.mediumId > cut_mu_id) | (events.Muon.tightId > cut_mu_id)) & (events.Muon.pfRelIso03_all < cut_mu_iso)]

        if self.region == 'D':
            # Reco event selection: minimal requirement for leptons
            #tau
            cut_tau_pt = 20. # Tau_pt > cut_tau_pt
            cut_tau_eta = 2.5 #abs(Tau_eta) < cut_tau_eta
            cut_tau_dz = 0.2 #abs(Tau_dz) < cut_tau_dz
            cut_tau_idVSmu = 4 # idDeepTau2018v2p5VSmu >= Tight
            cut_tau_idVSe = 3 # idDeepTau2018v2p5VSe >= VLoose
            cut_tau_idVSjet = 5 # idDeepTau2018v2p5VSjet >= Medium
            # + remove decay mode 5 and 6 as suggested here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
            events['SelTau'] = events.Tau[(events.Tau.pt > cut_tau_pt) & (np.abs(events.Tau.eta) < cut_tau_eta) & (np.abs(events.Tau.dz) < cut_tau_dz) & (events.Tau.idDeepTau2018v2p5VSmu >= cut_tau_idVSmu) & (events.Tau.idDeepTau2018v2p5VSe >= cut_tau_idVSe) & (events.Tau.idDeepTau2018v2p5VSjet >= cut_tau_idVSjet) & (events.Tau.decayMode != 5) & (events.Tau.decayMode != 6)]

            #electrons
            cut_e_pt = 25. # Electron_pt > cut_e_pt
            cut_e_eta = 2.5 # abs(Electron_eta) < cut_e_eta
            cut_e_dz = 0.2 #abs(Electron_dz) < cut_e_dz
            cut_e_dxy = 0.045 # abs(Electron_dxy) < cut_e_dxy
            cut_e_id = 0 # Electron_mvaIso_WP90 > cut_e_id (i.e True)
            events['SelElectron'] = events.Electron[(events.Electron.pt > cut_e_pt) & (np.abs(events.Electron.eta) < cut_e_eta) & (np.abs(events.Electron.dz) < cut_e_dz) & (np.abs(events.Electron.dxy) < cut_e_dxy) & (events.Electron.mvaIso_WP90 > cut_e_id)]

            #muons
            cut_mu_pt = 25. # Muon_pt > cut_mu_pt
            cut_mu_eta = 2.4 # abs(Muon_eta) < cut_mu_eta
            cut_mu_dz = 0.2 #abs(Muon_dz) < cut_mu_dz
            cut_mu_dxy = 0.045 # abs(Muon_dxy) < cut_mu_dxy
            cut_mu_id = 0 #(Muon_mediumId > cut_mu_id || Muon_tightId > cut_mu_id)
            cut_mu_iso = 0.15 # Muon_pfRelIso03_all < cut_mu_iso
            events['SelMuon'] = events.Muon[(events.Muon.pt > cut_mu_pt) & (np.abs(events.Muon.eta) < cut_mu_eta) & (np.abs(events.Muon.dz) < cut_mu_dz) & (np.abs(events.Muon.dxy) < cut_mu_dxy) & ((events.Muon.mediumId > cut_mu_id) | (events.Muon.tightId > cut_mu_id)) & (events.Muon.pfRelIso03_all < cut_mu_iso)]

        self.analyse_tmm(events, out, ds, mode)

        return out

    #helpful saved functions 
    def saved_leading_muon(self, events, Sel_Muon, out, ds):
        #print("save_muon " + ds)
        out[f'pt_mu1'].fill(ds=ds, pt=ak.flatten(Sel_Muon.pt, axis=None), weight=events.genWeight)
        out[f'eta_mu1'].fill(ds=ds, eta=ak.flatten(Sel_Muon.eta, axis=None), weight=events.genWeight)
        out[f'phi_mu1'].fill(ds=ds, phi=ak.flatten(Sel_Muon.phi, axis=None), weight=events.genWeight)
        out[f'mass_mu1'].fill(ds=ds, mass=ak.flatten(Sel_Muon.mass, axis=None), weight=events.genWeight)

    def saved_subleading_muon(self, events, Sel_Muon, out, ds):
        #print("save_muon " + ds)
        out[f'pt_mu2'].fill(ds=ds, pt=ak.flatten(Sel_Muon.pt, axis=None), weight=events.genWeight)
        out[f'eta_mu2'].fill(ds=ds, eta=ak.flatten(Sel_Muon.eta, axis=None), weight=events.genWeight)
        out[f'phi_mu2'].fill(ds=ds, phi=ak.flatten(Sel_Muon.phi, axis=None), weight=events.genWeight)
        out[f'mass_mu2'].fill(ds=ds, mass=ak.flatten(Sel_Muon.mass, axis=None), weight=events.genWeight)

    def saved_leading_electron(self, events, Sel_Electron, out, ds):
        #print("save_electron " + ds)
        out[f'pt_e1'].fill(ds=ds, pt=ak.flatten(Sel_Electron.pt, axis=None), weight=events.genWeight)
        out[f'eta_e1'].fill(ds=ds, eta=ak.flatten(Sel_Electron.eta, axis=None), weight=events.genWeight)
        out[f'phi_e1'].fill(ds=ds, phi=ak.flatten(Sel_Electron.phi, axis=None), weight=events.genWeight)
        out[f'mass_e1'].fill(ds=ds, mass=ak.flatten(Sel_Electron.mass, axis=None), weight=events.genWeight)

    def saved_subleading_electron(self, events, Sel_Electron, out, ds):
        #print("save_electron " + ds)
        out[f'pt_e2'].fill(ds=ds, pt=ak.flatten(Sel_Electron.pt, axis=None), weight=events.genWeight)
        out[f'eta_e2'].fill(ds=ds, eta=ak.flatten(Sel_Electron.eta, axis=None), weight=events.genWeight)
        out[f'phi_e2'].fill(ds=ds, phi=ak.flatten(Sel_Electron.phi, axis=None), weight=events.genWeight)
        out[f'mass_e2'].fill(ds=ds, mass=ak.flatten(Sel_Electron.mass, axis=None), weight=events.genWeight)

    def saved_leading_tau(self, events, Sel_Tau, out, ds):
        #print("save_tau " + ds)
        out[f'pt_tau1'].fill(ds=ds, pt=ak.flatten(Sel_Tau.pt, axis=None), weight=events.genWeight)
        out[f'eta_tau1'].fill(ds=ds, eta=ak.flatten(Sel_Tau.eta, axis=None), weight=events.genWeight)
        out[f'phi_tau1'].fill(ds=ds, phi=ak.flatten(Sel_Tau.phi, axis=None), weight=events.genWeight)
        out[f'mass_tau1'].fill(ds=ds, mass=ak.flatten(Sel_Tau.mass, axis=None), weight=events.genWeight)

    def saved_subleading_tau(self, events, Sel_Tau, out, ds):
        #print("save_tau " + ds)
        out[f'pt_tau2'].fill(ds=ds, pt=ak.flatten(Sel_Tau.pt, axis=None), weight=events.genWeight)
        out[f'eta_tau2'].fill(ds=ds, eta=ak.flatten(Sel_Tau.eta, axis=None), weight=events.genWeight)
        out[f'phi_tau2'].fill(ds=ds, phi=ak.flatten(Sel_Tau.phi, axis=None), weight=events.genWeight)
        out[f'mass_tau2'].fill(ds=ds, mass=ak.flatten(Sel_Tau.mass, axis=None), weight=events.genWeight)

    def saved_subsubleading_tau(self, events, Sel_Tau, out, ds):
        #print("save_tau " + ds)
        out[f'pt_tau3'].fill(ds=ds, pt=ak.flatten(Sel_Tau.pt, axis=None), weight=events.genWeight)
        out[f'eta_tau3'].fill(ds=ds, eta=ak.flatten(Sel_Tau.eta, axis=None), weight=events.genWeight)
        out[f'phi_tau3'].fill(ds=ds, phi=ak.flatten(Sel_Tau.phi, axis=None), weight=events.genWeight)
        out[f'mass_tau3'].fill(ds=ds, mass=ak.flatten(Sel_Tau.mass, axis=None), weight=events.genWeight)

    def saved_dilepton_mass(self, events, Lepton1, Lepton2, out, ds):
        #print("dilepton_mass " + ds)
        out[f'comb_mass_l1l2'].fill(ds=ds, mc=ak.flatten((Lepton1 + Lepton2).mass, axis=None), weight=events.genWeight)

    def saved_dilepton_mass_taul1_OS(self, events, Lepton, Tau1, Tau2, out, ds):
        # +++ and --- events not recorded!

        #Lepton and Tau1 OS (and tau2 SS as Lepton)
        sel1 = ak.flatten((Tau1.charge != Tau2.charge) & (Lepton.charge != Tau1.charge))
        out[f'comb_mass_taul1'].fill(ds=ds, mc=ak.flatten((Lepton[sel1] + Tau1[sel1]).mass, axis=None), weight=events[sel1].genWeight)
        #Lepton and Tau2 OS (and tau1 SS as Lepton)
        sel2 = ak.flatten((Tau1.charge != Tau2.charge) & (Lepton.charge != Tau2.charge))
        out[f'comb_mass_taul1'].fill(ds=ds, mc=ak.flatten((Lepton[sel2] + Tau2[sel2]).mass, axis=None), weight=events[sel2].genWeight)
        #Lepton and Tau2/Tau1 OS --> take the leading tau
        sel3 = ak.flatten((Tau1.charge == Tau2.charge) & (Lepton.charge != Tau2.charge) & (Tau1.pt >= Tau2.pt))
        out[f'comb_mass_taul1'].fill(ds=ds, mc=ak.flatten((Lepton[sel3] + Tau1[sel3]).mass, axis=None), weight=events[sel3].genWeight)
        sel4 = ak.flatten((Tau1.charge == Tau2.charge) & (Lepton.charge != Tau2.charge) & (Tau1.pt < Tau2.pt))
        out[f'comb_mass_taul1'].fill(ds=ds, mc=ak.flatten((Lepton[sel4] + Tau2[sel4]).mass, axis=None), weight=events[sel4].genWeight)

        #check orthogonality/sanity check
        sel5 = ak.flatten((Tau1.charge == Tau2.charge) & (Lepton.charge == Tau2.charge))
        if (len(events) - (ak.sum(sel1)+ak.sum(sel2)+ak.sum(sel3)+ak.sum(sel4)+ak.sum(sel5))) != 0 :
            print('problem with saved_dilepton_mass_taul1_OS for ds:' + ds)

    def saved_drl1l2(self, events, Lepton1, Lepton2, out, ds):
        #print("save_drl1l2 " + ds)
        out[f'dr_l1l2'].fill(ds=ds, dr=ak.flatten(delta_r(Lepton1,Lepton2), axis=None), weight=events.genWeight)

    def saved_MET(self, events, out, ds):
        #print("save_drl1l2 " + ds)
        out[f'met'].fill(ds=ds, met=events.MET.pt, weight=events.genWeight)

    def saved_pt_sum_l1l2l3(self, events, Lepton1, Lepton2, Lepton3, out, ds):
        out[f'pt_sum_l1l2l3'].fill(ds=ds, pt=ak.flatten((Lepton1.pt + Lepton2.pt  + Lepton3.pt), axis=None), weight=events.genWeight)

    def saved_pt_sum_l1l2MET(self, events, Lepton1, Lepton2, out, ds):
        out[f'pt_sum_l1l2MET'].fill(ds=ds, pt=ak.flatten((Lepton1.pt + Lepton2.pt  + events.MET.pt), axis=None), weight=events.genWeight)

    def saved_mT_tautau(self, events, Lepton1, Lepton2, out, ds):
        mT_tautau = np.sqrt( Lepton1.mass**2 + Lepton2.mass**2 + 2*(np.sqrt(Lepton1.mass**2 + Lepton1.pt**2)*np.sqrt(Lepton2.mass**2 + Lepton2.pt**2) - Lepton1.pt*Lepton2.pt*np.cos(abs(Lepton1.phi - Lepton2.phi))))
        out[f'mT_tautau'].fill(ds=ds, mt=ak.flatten(mT_tautau, axis=None), weight=events.genWeight)

    def saved_mT_l1MET(self, events, Lepton1, MET, out, ds):
        mT_l1MET = np.sqrt( Lepton1.mass**2 + 2*(np.sqrt(Lepton1.mass**2 + Lepton1.pt**2)*MET.pt - Lepton1.pt*MET.pt*np.cos(abs(Lepton1.phi - MET.phi))))
        out[f'mT_l1MET'].fill(ds=ds, mt=ak.flatten(mT_l1MET, axis=None), weight=events.genWeight)

    def analyse_tmm(self, events, out, ds, mode):

        # select tmm events: require 2 reco mu and 1 reco tau 
        events_tmm = events[(ak.num(events.SelMuon) == 2) & (ak.num(events.SelTau) == 1)]

        out[f'sumw_3leptons'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_3leptons'][ds] += len(events_tmm)

        # events should pass most efficient HLT (for now only 1: IsoMu24)
        events_tmm = events_tmm[events_tmm.HLT.IsoMu24]

        out[f'sumw_hlt'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_hlt'][ds] += len(events_tmm)

        if len(events_tmm) == 0:
            return

        # select reco muon that match HLT IsoMu24
        Sel_Muon = self.select_lep1_IsoMu24(events_tmm, min_dr_cut=0.2)

        #removing non matching events
        events_tmm = events_tmm[(ak.num(Sel_Muon) == 1)]
        Sel_Muon = Sel_Muon[(ak.num(Sel_Muon) == 1)]

        out[f'sumw_l1sel'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_l1sel'][ds] += len(events_tmm)

        # select reco Muon2 with dr(mu2,mu)>0.5 (in case there is more than 1 selected we choose the one with higher pt )
        Sel_Muon2 = self.select_lep2(events_tmm, Sel_Muon, type='muon', delta_r_cut = 0.5)

        #removing non matching events
        Sel_Muon = Sel_Muon[(ak.num(Sel_Muon2) == 1)]
        events_tmm = events_tmm[(ak.num(Sel_Muon2) == 1)]
        Sel_Muon2 = Sel_Muon2[(ak.num(Sel_Muon2) == 1)]

        out[f'sumw_l2sel'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_l2sel'][ds] += len(events_tmm)

        # select reco Tau with dr(tau,mu)>0.5 and dr(tau,muon2)>0.5 (in case there is more than 1 selected we choose the one with higher pt )
        Sel_Tau = self.select_lep3(events_tmm, Sel_Muon, Sel_Muon2, type='tau', delta_r_cut = 0.5)

        #removing non matching events
        Sel_Muon = Sel_Muon[(ak.num(Sel_Tau) == 1)]
        Sel_Muon2 = Sel_Muon2[(ak.num(Sel_Tau) == 1)]
        events_tmm = events_tmm[(ak.num(Sel_Tau) == 1)]
        Sel_Tau = Sel_Tau[(ak.num(Sel_Tau) == 1)]
    
        out[f'sumw_l3sel'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_l3sel'][ds] += len(events_tmm)
        '''
        events_tmm, Sel_Tau, Sel_Muon, Sel_Muon2 = self.bjet_veto(events_tmm, Sel_Tau, Sel_Muon, Sel_Muon2)
        out[f'sumw_bjetveto'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_bjetveto'][ds] += len(events_tmm)

        events_tmm, Sel_Tau, Sel_Muon, Sel_Muon2 = self.charge_veto(events_tmm, Sel_Tau, Sel_Muon, Sel_Muon2)
        out[f'sumw_chargeveto'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_chargeveto'][ds] += len(events_tmm)

        events_tmm, Sel_Tau, Sel_Muon, Sel_Muon2 = self.met_veto(events_tmm, Sel_Tau, Sel_Muon, Sel_Muon2)
        out[f'sumw_metselection'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_metselection'][ds] += len(events_tmm)

        events_tmm, Sel_Tau, Sel_Muon, Sel_Muon2 = self.z_veto_tll(events_tmm, Sel_Tau, Sel_Muon, Sel_Muon2)
        out[f'sumw_zveto'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_zveto'][ds] += len(events_tmm)
        '''
        #computing corrections
        if mode != 'Data':
            pileup_corr = get_pileup_correction(events_tmm, 'nominal')* self.norm_factor
            sf_mu1 = compute_sf_mu(Sel_Muon)
            Trigger_eff_corr_mu = get_trigger_correction_mu(Sel_Muon)
            sf_mu2 = compute_sf_mu(Sel_Muon2)
            # as we use DeepTau2018v2p5, we don't have corrections yet
            sf_tau = compute_sf_tau_e(Sel_Tau)
            events_tmm.genWeight = events_tmm.genWeight * sf_mu1 * Trigger_eff_corr_mu * sf_mu2 * sf_tau * pileup_corr

        out[f'sumw_corrections'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_corrections'][ds] += len(events_tmm)

        # Save histograms
        self.saved_leading_muon(events_tmm, Sel_Muon, out, ds)
        self.saved_subleading_muon(events_tmm, Sel_Muon2, out, ds)
        self.saved_leading_tau(events_tmm, Sel_Tau, out, ds)
        self.saved_dilepton_mass(events_tmm, Sel_Muon, Sel_Muon2, out, ds)
        self.saved_MET(events_tmm, out, ds)
        self.saved_drl1l2(events_tmm, Sel_Muon, Sel_Muon2, out, ds)

        return events_tmm

    #helpful selection functions 
    def select_lep1_IsoMu24(self, events, min_dr_cut = 0.2):
        # select reco muon that match HLT
        # Trigger muon is a mu (id == 13),abs(eta) < 2.4 and pt > 25 (HLT marge) with  TrigObj_filterBits for Muon: 2 = Iso and 8 = 1mu
        Trigger_Muon = events.TrigObj[ (abs(events.TrigObj.id) == 13) & (abs(events.TrigObj.eta) < 2.4) & (events.TrigObj.pt > 25.) & ((events.TrigObj.filterBits & (2+8)) != 0)]
        #We need only one trigger object: in case there is more, we choose the one with higher pt 
        Trigger_Muon = Trigger_Muon[ak.argmax(Trigger_Muon.pt, axis=-1, keepdims = True)] 
        #Trigger_Muon = ak.fill_none(Trigger_Muon, [])

        # reco muon 
        Reco_Muon = events.SelMuon

        #matching: select the reco muon with min dr wrt the trigger muon + impose minimum dr 
        trigger_muon, reco_muon = ak.unzip(ak.cartesian([Trigger_Muon, Reco_Muon], nested=True))
        Sel_Muon = reco_muon[ak.argmin(delta_r(trigger_muon, reco_muon), axis=-1, keepdims = True , mask_identity = True)] # take the one with min dr
        Sel_Muon = Sel_Muon[:,0]

        cut_dr_mask = delta_r(Sel_Muon, Trigger_Muon) < min_dr_cut # remove too high dr matching
        Sel_Muon = Sel_Muon[ cut_dr_mask ]

        return Sel_Muon

    def select_lep1_Ele32_WPTight_Gsf_L1DoubleEG(self, events, min_dr_cut = 0.2):
        # select reco electron that match HLT
        # Trigger e is a e (id == 11),abs(eta) < 2.5 and pt > 33 (HLT marge) with  TrigObj_filterBits for Electron: 2 = 1e WPTight
        Trigger_Electron = events.TrigObj[ (abs(events.TrigObj.id) == 11) & (abs(events.TrigObj.eta) < 2.5) & (events.TrigObj.pt > 33.) & ((events.TrigObj.filterBits & (2)) != 0)]  
        #We need only one trigger object: in case there is more, we choose the one with higher pt 
        Trigger_Electron = Trigger_Electron[ak.argmax(Trigger_Electron.pt, axis=-1, keepdims = True)] 
        
        # reco electron 
        Reco_Electron = events.SelElectron

        #matching: select the reco electron with min dr wrt the trigger electron + impose minimum dr 
        trigger_electron, reco_electron = ak.unzip(ak.cartesian([Trigger_Electron, Reco_Electron], nested=True))
        Sel_Electron = reco_electron[ak.argmin(delta_r(trigger_electron, reco_electron), axis=-1, keepdims = True)] # take the one with min dr
        Sel_Electron = Sel_Electron[:,0]

        cut_dr_mask = delta_r(Sel_Electron, Trigger_Electron) < min_dr_cut # remove too high dr matching
        Sel_Electron = Sel_Electron[ cut_dr_mask ]

        return Sel_Electron

    def select_lep1_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_old(self, events, min_dr_cut=0.2):

        # select reco tau that match HLT
        # Trigger tau is a tau (id == 15),abs(eta) < 2.2 and pt > 0 (HLT) with  TrigObj_filterBits for Tau: 2 = MediumChargedIso, 16 = HPS, 64 = di-tau
        Trigger_Tau = events.TrigObj[ (abs(events.TrigObj.id) == 15) & (abs(events.TrigObj.eta) < 2.2) & ((events.TrigObj.filterBits & (2+16+64)) != 0)] 
        #We need only one trigger object: in case there is more, we choose the one with higher pt
        Trigger_Tau = Trigger_Tau[ak.argmax(Trigger_Tau.pt, axis=-1, keepdims = True)] 

        # reco electron 
        Reco_Tau = events.SelTau

        #matching: select the reco tau with min dr wrt the trigger Tau + impose minimum dr 
        trigger_tau, reco_tau = ak.unzip(ak.cartesian([Trigger_Tau, Reco_Tau], nested=True))
        Sel_Tau = reco_tau[ak.argmin(delta_r(trigger_tau, reco_tau), axis=-1, keepdims = True)] # take the one with min dr
        Sel_Tau = Sel_Tau[:,0]

        cut_dr_mask = delta_r(Sel_Tau, Trigger_Tau) < min_dr_cut # remove too high dr matching
        Sel_Tau = Sel_Tau[cut_dr_mask]
        
        return Sel_Tau

    def select_lep_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg(self, events, min_dr_cut=0.2, min_pt_cut=40.):
        # select 2 reco tau that match HLT

        # Trigger tau is a tau (id == 15),abs(eta) < 2.2 and pt > 0 (HLT) with  TrigObj_filterBits for Tau: 2 = MediumChargedIso, 16 = HPS, 64 = di-tau
        Trigger_Tau = events.TrigObj[ (abs(events.TrigObj.id) == 15) & (abs(events.TrigObj.eta) < 2.2) & ((events.TrigObj.filterBits & (2+16+64)) != 0)] 

        #We need at least two trigger object: in case there is less, we remove it
        cut = (ak.num(Trigger_Tau) >= 2)
        events = events[cut]
        Trigger_Tau = Trigger_Tau[cut]

        # reco electron 
        Reco_Tau = events.SelTau

        #matching l1: select the reco tau with min dr wrt the trigger Tau
        # for each reco tau, we select the trigger tau with min dr
        reco_tau, trigger_tau = ak.unzip(ak.cartesian([Reco_Tau, Trigger_Tau], nested=True))
        Sel_Tau = reco_tau[ak.argmin(delta_r(trigger_tau, reco_tau), axis=-1, keepdims = True)]
        Sel_Trigger_Tau = trigger_tau[ak.argmin(delta_r(trigger_tau, reco_tau), axis=-1, keepdims = True)] 
        #then we take min dr between reco and trigger overall the best matching trigger tau
        Sel_Trigger_Tau1 = Sel_Trigger_Tau[ak.argmin(delta_r(Sel_Tau, Sel_Trigger_Tau), axis=1, keepdims = False)][:,0]
        Sel_Tau1 = Sel_Tau[ak.argmin(delta_r(Sel_Tau, Sel_Trigger_Tau), axis=1, keepdims = False)][:,0]

        #here we can require a min dr between trigger and reco

        #remove Sel_Trigger_Tau1 from the list of trigger object
        trigger_tau, sel_trigger_tau1 = ak.unzip(ak.cartesian([Trigger_Tau, Sel_Trigger_Tau1], nested=False))
        Trigger_Tau = Trigger_Tau[delta_r(trigger_tau, sel_trigger_tau1) >= 0.02]

        #We need at least one trigger object: in case there is less, we remove it
        cut = (ak.num(Trigger_Tau) >= 1)
        events = events[cut]
        Trigger_Tau = Trigger_Tau[cut]

        #remove Sel_Tau1 from the list of reco object
        reco_tau, sel_tau1 = ak.unzip(ak.cartesian([Reco_Tau, Sel_Tau1], nested=False))
        Reco_Tau = Reco_Tau[delta_r(reco_tau, sel_tau1) >=0.02]

        #We need at least one reco object: in case there is less, we remove it
        cut = (ak.num(Reco_Tau) >= 1)
        events = events[cut]
        Reco_Tau = Reco_Tau[cut]

        #matching l2: select the reco tau with min dr wrt the remaining trigger Tau
        # for each reco tau remaining, we select the trigger tau with min dr
        reco_tau, trigger_tau = ak.unzip(ak.cartesian([Reco_Tau, Trigger_Tau], nested=True))
        Sel_Tau = reco_tau[ak.argmin(delta_r(trigger_tau, reco_tau), axis=-1, keepdims = True)]
        Sel_Trigger_Tau = trigger_tau[ak.argmin(delta_r(trigger_tau, reco_tau), axis=-1, keepdims = True)] 
        #then we take min dr between reco and trigger overall the best matching trigger tau
        Sel_Trigger_Tau2 = Sel_Trigger_Tau[ak.argmin(delta_r(Sel_Tau, Sel_Trigger_Tau), axis=1, keepdims = False)][:,0]
        Sel_Tau2 = Sel_Tau[ak.argmin(delta_r(Sel_Tau, Sel_Trigger_Tau), axis=1, keepdims = False)][:,0]

        #here we can require a min dr between trigger and reco

        # impose min dr between l1 and l2
        cut_dr_mask = (delta_r(Sel_Tau1, Sel_Tau2) >= min_dr_cut)[:,0]
        Sel_Tau1 = Sel_Tau1[cut_dr_mask]
        Sel_Tau2 = Sel_Tau2[cut_dr_mask]
        events = events[cut_dr_mask]

        # impose min pt > 40 GeV for Tau1 and Tau 2
        cut_pt_mask = ak.any((Sel_Tau1.pt > min_pt_cut) & (Sel_Tau2.pt > min_pt_cut), axis=1)
        Sel_Tau1 = Sel_Tau1[cut_pt_mask]
        Sel_Tau2 = Sel_Tau2[cut_pt_mask]
        events = events[cut_pt_mask]
        
        return events, Sel_Tau1, Sel_Tau2

    def select_lep2(self, events, Sel_Lep1, type, delta_r_cut = 0.5):

        if type not in ['tau','muon', 'electron']:
            raise 'Unvalid type: must be: tau,muon or electron'

        if type == 'tau':
            Reco_lep2 = events.SelTau

        if type == 'muon':
            Reco_lep2 = events.SelMuon

        if type == 'electron':
            Reco_lep2 = events.SelElectron

        # select reco lep2 with dr(lep1,lep2)>0.5 
        sel_lep1, reco_lep2= ak.unzip(ak.cartesian([Sel_Lep1, Reco_lep2], nested=True))
        Sel_Lep2 = reco_lep2[(delta_r(sel_lep1,reco_lep2)> delta_r_cut)]

        #We need only one tau: in case there is more, we choose the one with higher pt 
        Sel_Lep2 = Sel_Lep2[ak.max(Sel_Lep2.pt, axis=-1) == Sel_Lep2.pt]
        Sel_Lep2 = ak.fill_none(Sel_Lep2[:,0], [], axis=0)

        return Sel_Lep2

    def select_lep3(self, events, Sel_Lep1, Sel_Lep2, type, delta_r_cut = 0.5):

        if type not in ['tau','muon', 'electron']:
            raise 'Unvalid type: must be: tau,muon or electron'

        if type == 'tau':
            Reco_lep3 = events.SelTau

        if type == 'muon':
            Reco_lep3 = events.SelMuon

        if type == 'electron':
            Reco_lep3 = events.SelElectron

        # select reco lep3 that is not the same as the lep1 and lep2
        sel_lep1, reco_lep3 = ak.unzip(ak.cartesian([Sel_Lep1, Reco_lep3], nested=True))
        match1 = (delta_r(sel_lep1,reco_lep3)> delta_r_cut)

        sel_lep2, reco_lep3 = ak.unzip(ak.cartesian([Sel_Lep2, Reco_lep3], nested=True))
        match2 = (delta_r(sel_lep2,reco_lep3)> delta_r_cut)
        Sel_Lep3 = reco_lep3[match1 & match2]

        #We need only one l3: in case there is more, we choose the one with higher pt 
        Sel_Lep3 = Sel_Lep3[ak.max(Sel_Lep3.pt, axis=-1) == Sel_Lep3.pt]
        Sel_Lep3 = ak.fill_none(Sel_Lep3[:,0], [], axis=0)

        return Sel_Lep3

    def bjet_veto(self, events, Lepton1, Lepton2, Lepton3, delta_r_cut = 0.5):
        # Reject events in order to reduce the background from processes with top quarks (producing jets)
        # the dr non-matching with lepton are here to make sure jet is not misidentify as a signal lepton
        bjets_candidates = events.Jet[(events.Jet.pt > 20.) & (events.Jet.eta < 2.5) & (events.Jet.jetId >= 4) & (events.Jet.btagDeepFlavB > 0.0490)]

        sel_lep1, sel_jets = ak.unzip(ak.cartesian([Lepton1, bjets_candidates], nested=True))
        drcut_jetslep1 = (delta_r(sel_lep1,sel_jets) > delta_r_cut)

        sel_lep2, sel_jets = ak.unzip(ak.cartesian([Lepton2, bjets_candidates], nested=True))
        drcut_jetslep2 = (delta_r(sel_lep2,sel_jets) > delta_r_cut)

        sel_lep3, sel_jets = ak.unzip(ak.cartesian([Lepton3, bjets_candidates], nested=True))
        drcut_jetslep3 = (delta_r(sel_lep3,sel_jets) > delta_r_cut)

        nonmatching_lep_jets = sel_jets[drcut_jetslep1 & drcut_jetslep2 & drcut_jetslep3]
        nonmatching_lep_jets = ak.fill_none(nonmatching_lep_jets[:,0], [], axis=0)

        cut = (ak.num(nonmatching_lep_jets) == 0)
        events = events[cut]
        Lepton1 = Lepton1[cut]
        Lepton2 = Lepton2[cut]
        Lepton3 = Lepton3[cut]
        return events, Lepton1, Lepton2, Lepton3

    def charge_veto(self, events, Lepton1, Lepton2, Lepton3):
        # remove +++ and --- events
        cut = ak.flatten( (Lepton1.charge != Lepton2.charge) | (Lepton1.charge != Lepton3.charge))
        events = events[cut]
        Lepton1 = Lepton1[cut]
        Lepton2 = Lepton2[cut]
        Lepton3 = Lepton3[cut]
        return events, Lepton1, Lepton2, Lepton3

    def met_veto(self, events, Lepton1, Lepton2, Lepton3):
        metcut = 25.
        cut = events.MET.pt > metcut
        events = events[cut]
        Lepton1 = Lepton1[cut]
        Lepton2 = Lepton2[cut]
        Lepton3 = Lepton3[cut]
        return events, Lepton1, Lepton2, Lepton3

    def z_veto_tll(self, events, Tau1, Lepton1, Lepton2):
        # work for tee tmm and tem_OS channel (for tem_SS just invert Tau1 with l1 or l2 as we removed +++ --- events)
        mass_z = 91.2 #GeV
        interval = 10.
        cut = ak.flatten( ((Lepton1 + Lepton2).mass < (mass_z - interval)) |  ((Lepton1 + Lepton2).mass > (mass_z + interval)) )
        events = events[cut]
        Tau1 = Tau1[cut]
        Lepton1 = Lepton1[cut]
        Lepton2 = Lepton2[cut]
        return events, Tau1, Lepton1, Lepton2

    def z_veto_ttl(self, events, Tau1, Tau2, Lepton):
        # work for tte ttm 
        mass_z = 91.2 #GeV
        interval = 10.
        cut_1 = ak.flatten( ((Lepton + Tau1).mass < (mass_z - interval)) |  ((Lepton + Tau1).mass > (mass_z + interval)) )
        cut_2 = ak.flatten( ((Lepton + Tau2).mass < (mass_z - interval)) |  ((Lepton + Tau2).mass > (mass_z + interval)) )

        sel1 = ak.flatten((Tau1.charge != Tau2.charge) & (Lepton.charge != Tau1.charge)) # +-- or -++ --> we take lep tau1 dimass
        sel2 = ak.flatten((Tau1.charge != Tau2.charge) & (Lepton.charge != Tau2.charge)) # +-+ or -+- --> we take lep tau2 dimass
        sel3 = ak.flatten((Tau1.charge == Tau2.charge) & (Lepton.charge != Tau2.charge) & (Tau1.pt >= Tau2.pt)) # ++- or --+ and tau1 has higher pt than tau2 --> we take lep tau1 dimass
        sel4 = ak.flatten((Tau1.charge == Tau2.charge) & (Lepton.charge != Tau2.charge) & (Tau1.pt < Tau2.pt)) # ++- or --+ and tau2 has higher pt than tau1 --> we take lep tau2 dimass

        cut = (sel1 & cut_1) | (sel2 & cut_2) | (sel3 & cut_1) | (sel4 & cut_1)
        events = events[cut]
        Tau1 = Tau1[cut]
        Tau2 = Tau2[cut]
        Lepton = Lepton[cut]
        return events, Tau1, Tau2, Lepton

    def postprocess(self, accumulator):
        return accumulator
