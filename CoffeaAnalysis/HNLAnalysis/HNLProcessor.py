import numpy as np
import awkward as ak

from CoffeaAnalysis.HNLAnalysis.helpers import apply_golden_run, apply_reweight, apply_MET_Filter
from CoffeaAnalysis.HNLAnalysis.correction_helpers import compute_tau_e_corr, compute_electron_ES_corr, compute_electron_ER_corr, compute_jet_corr, MET_correction

class HNLProcessor():
    def __init__(self, stitched_list, tag, xsecs, periods, dataHLT, debugMode):
        #define global variables
        if stitched_list is None or len(stitched_list) == 0:
            raise 'Missing stitched_list in samples_{era}.yaml'
        self.stitched_list = stitched_list

        if tag is None or len(tag) == 0:
            raise 'Missing tag'
        self.tag = tag

        self.xsecs = xsecs

        self.period = periods

        if self.period in ['2018','2017','2016','2016_HIPM']:
            self.DeepTauVersion = 'DeepTau2018v2p5'
        else:
            self.DeepTauVersion = 'DeepTau2017v2p1'
        
        self.dataHLT = dataHLT

        self.debugMode = debugMode
    
    # Init process
    def init_process(self, out, events):

        ds = events.metadata["dataset"] # dataset name
        ds = ds.split('_nano')[0]
            
        if self.dataHLT in ds:
            events['genWeight'] = events.run > 0 #setup genWeight for data (1.)
        else:
            events["Jet","MatchedGenPt"] = events.Jet.matched_gen.pt # save this info for JET correction

        if self.debugMode:
            events = events[0:5000]
            print(f'Processing: {ds} ({len(events)} events in DEBUGMODE)')
        else:
            print(f'Processing: {ds} ({len(events)} events)')

        # Save initial SumGenWeights 
        out['sumw_init'][ds] += ak.sum(events.genWeight)
        out['n_ev_init'][ds] += len(events)

        # defining the mode
        mode ='MCbackground' # default mode

        if self.dataHLT in ds:
            mode ='Data'
            # only keep the "golden" runs
            events = apply_golden_run(ds, events, self.period)
            #A,B,C and D together
            ds = ds[0:-1]

        if 'HNL' in ds:
            mode ='signal'
        
        #make ds and mode global var
        self.ds = ds
        self.mode = mode
        print(f'Analysis in mode {self.mode}')

        if self.mode != 'Data':
            #check if Xsec is missing
            if self.xsecs is None:
                raise 'Missing xsecs'
            #reweights events with lumi x xsec / n_events (with PU correction) + applying stitching weights for DY and WJets samples 
            events = apply_reweight(self.ds, events, self.stitched_list, self.dataHLT, self.xsecs, self.period)

        # sumw after reweighting (for MC) and "golden" runs selection (for data)
        out['sumw_reweight'][self.ds] += ak.sum(events.genWeight)
        out['n_ev_reweight'][self.ds] += len(events)

        # MET filters
        events = apply_MET_Filter(events, self.period)

        # sumw after application of MET filters
        out['sumw_MET_Filter'][self.ds] += ak.sum(events.genWeight)
        out['n_ev_MET_Filter'][self.ds] += len(events)

        # Reco event selection: common minimal requirement for leptons
        #tau 
        self.cut_tau_pt = 20. # Tau_pt > cut_tau_pt (general recommendations)
        self.cut_tau_eta = 2.5 #abs(Tau_eta) < cut_tau_eta (general recommendations for DeepTau2p5: 2.3 for DeepTau2p1)
        self.cut_tau_dz = 0.2 #abs(Tau_dz) < cut_tau_dz
        self.cut_tau_idVSe = 2 # idDeepTauVSe >= VVLoose
        self.cut_tau_idVSjet = 2 # idDeepTauVSjet >= VVLoose
        self.cut_tau_idVSmu = 4 # idDeepTauVSmu >= Tight

        #electrons
        self.cut_e_pt = 10. # Electron_pt > cut_e_pt
        self.cut_e_eta = 2.5 # abs(Electron_eta) < cut_e_eta
        self.cut_e_dz = 0.2 #abs(Electron_dz) < cut_e_dz
        self.cut_e_dxy = 0.045 # abs(Electron_dxy) < cut_e_dxy
        self.cut_e_iso = 0.4 # Electron_pfRelIso03_all < cut_e_iso

        #muons
        self.cut_mu_pt = 15. # Muon_pt > cut_mu_pt
        self.cut_mu_eta = 2.4 # abs(Muon_eta) < cut_mu_eta
        self.cut_mu_dz = 0.2 #abs(Muon_dz) < cut_mu_dz
        self.cut_mu_dxy = 0.045 # abs(Muon_dxy) < cut_mu_dxy
        self.cut_mu_iso = 0.4 # Muon_pfRelIso03_all < cut_mu_iso

        return events, out

    def Lepton_selection(self, events, Treename = None):
        #tau
        # cuts in HNLProcessor + remove decay mode 5 and 6 as suggested here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
        events['SelTau'] = events.Tau[(np.abs(events.Tau.eta) < self.cut_tau_eta) & (np.abs(events.Tau.dz) < self.cut_tau_dz) & (events.Tau[f'id{self.DeepTauVersion}VSmu'] >= self.cut_tau_idVSmu) & (events.Tau[f'id{self.DeepTauVersion}VSe'] >= self.cut_tau_idVSe) & (events.Tau[f'id{self.DeepTauVersion}VSjet'] >= self.cut_tau_idVSjet) & (events.Tau.decayMode != 5) & (events.Tau.decayMode != 6)]
        #electrons
        # cuts in HNLProcessor + mvaNoIso_WP90 > 0 (i.e True)
        events['SelElectron'] = events.Electron[(np.abs(events.Electron.eta) < self.cut_e_eta) & (np.abs(events.Electron.dz) < self.cut_e_dz) & (np.abs(events.Electron.dxy) < self.cut_e_dxy) & (events.Electron.mvaNoIso_WP90 > 0) & (events.Electron.pfRelIso03_all < self.cut_e_iso)]
        #muons
        # cuts in HNLProcessor + Muon_mediumId
        events['SelMuon'] = events.Muon[(events.Muon.pt > self.cut_mu_pt) & (np.abs(events.Muon.eta) < self.cut_mu_eta) & (np.abs(events.Muon.dz) < self.cut_mu_dz) & (np.abs(events.Muon.dxy) < self.cut_mu_dxy) & (events.Muon.mediumId > 0) & (events.Muon.pfRelIso03_all < self.cut_mu_iso)]
        #jets
        events['SelJet'] = events.Jet
        #MET
        events['SelMET'] = events.MET

        #Apply JEC/JER corr and Jet veto
        corrected_jets = compute_jet_corr(events, self.period, self.mode)

        #apply energy corrections:
        if self.mode != 'Data':
            tau_es, tau_es_up, tau_es_down = compute_tau_e_corr(events.SelTau, self.period)
            Tau_not_corrected = events.SelTau
            if Treename == None:
                events["SelTau","pt"] = events.SelTau.pt*tau_es
                events["SelTau","mass"] = events.SelTau.mass*tau_es
                #MET correction due to tau variation
                corrected_met = MET_correction(events.SelMET, Tau_not_corrected, events.SelTau) 
                #MET correction due to jet variation
                corrected_met = MET_correction(corrected_met, events.SelJet, corrected_jets) 
                events["SelJet","pt"] = corrected_jets.pt
                events["SelJet","mass"] = corrected_jets.mass
                events["SelMET","pt"] = corrected_met.pt
                events["SelMET","phi"] = corrected_met.phi
                
            else:
                lst = Treename.split('_')
                if lst[1] in ['GenuineTauES', 'GenuineElectronES', 'GenuineMuonES']:
                    if lst[1] == 'GenuineTauES':
                        mask_genPartFlav = (events.SelTau.genPartFlav == 5)
                        if lst[2] == 'DM0':
                            mask = (events.SelTau.decayMode == 0) & mask_genPartFlav
                        if lst[2] == 'DM1':
                            mask = (events.SelTau.decayMode == 1) & mask_genPartFlav
                        if lst[2] == '3prong':
                            mask = ((events.SelTau.decayMode == 10) | (events.SelTau.decayMode == 11)) & mask_genPartFlav
                    
                    if lst[1] == 'GenuineElectronES':
                        mask_genPartFlav = (events.SelTau.genPartFlav == 1) | (events.SelTau.genPartFlav == 3)
                        if lst[2] == 'DM0':
                            mask = (events.SelTau.decayMode == 0) & mask_genPartFlav
                        if lst[2] == 'DM1':
                            mask = (events.SelTau.decayMode == 1) & mask_genPartFlav
                        if lst[2] == '3prong':
                            mask = ((events.SelTau.decayMode == 10) | (events.SelTau.decayMode == 11)) & mask_genPartFlav
                    
                    if lst[1] == 'GenuineMuonES':
                        mask = (events.SelTau.genPartFlav == 2) | (events.SelTau.genPartFlav == 4)
                    
                    if lst[-1] == 'up':
                        sf = ak.where(mask,tau_es_up, tau_es)
                    if lst[-1] == 'down':
                        sf = ak.where(mask,tau_es_down, tau_es)

                    events["SelTau","pt"] = events.SelTau.pt*sf
                    events["SelTau","mass"] = events.SelTau.mass*sf
                    #MET correction due to tau variation
                    corrected_met = MET_correction(events.SelMET, Tau_not_corrected, events.SelTau)
                    #MET correction due to jet variation
                    corrected_met = MET_correction(corrected_met, events.SelJet, corrected_jets) 
                    events["SelJet","pt"] = corrected_jets.pt
                    events["SelJet","mass"] = corrected_jets.mass
                    events["SelMET","pt"] = corrected_met.pt
                    events["SelMET","phi"] = corrected_met.phi

                if lst[1] in ['ElectronES', 'ElectronER']:
                    if lst[1] == 'ElectronES':
                        ES_up, ES_down = compute_electron_ES_corr(events.SelElectron)
                    if lst[1] == 'ElectronER':
                        ES_up, ES_down = compute_electron_ER_corr(events.SelElectron)

                    Electron_not_corrected = events.SelElectron
                    if lst[-1] == 'up':
                        events["SelElectron","pt"] = events.SelElectron.pt * ES_up
                        events["SelElectron","mass"] = events.SelElectron.mass * ES_up

                    if lst[-1] == 'down':
                        events["SelElectron","pt"] = events.SelElectron.pt * ES_down
                        events["SelElectron","mass"] = events.SelElectron.mass * ES_down

                    #MET correction due to e variation
                    corrected_met = MET_correction(events.SelMET, Electron_not_corrected, events.SelElectron)
                    events["SelTau","pt"] = events.SelTau.pt*tau_es
                    events["SelTau","mass"] = events.SelTau.mass*tau_es
                    #MET correction due to tau variation
                    corrected_met = MET_correction(corrected_met, Tau_not_corrected, events.SelTau)
                    #MET correction due to jet variation
                    corrected_met = MET_correction(corrected_met, events.SelJet, corrected_jets) 
                    events["SelJet","pt"] = corrected_jets.pt
                    events["SelJet","mass"] = corrected_jets.mass
                    events["SelMET","pt"] = corrected_met.pt
                    events["SelMET","phi"] = corrected_met.phi

                if lst[1] in ['JES', 'JER']:
                    if lst[1] == 'JES':
                        corr_jets = corrected_jets.JES_jes
                    if lst[1] == 'JER':
                        corr_jets = corrected_jets.JER

                    Jet_not_corrected = events.SelJet
                    if lst[-1] == 'up':
                        events["SelJet","pt"] = corr_jets.up.pt
                        events["SelJet","mass"] = corr_jets.up.mass
                    if lst[-1] == 'down':
                        events["SelJet","pt"] = corr_jets.down.pt
                        events["SelJet","mass"] = corr_jets.down.mass

                    events["SelTau","pt"] = events.SelTau.pt*tau_es
                    events["SelTau","mass"] = events.SelTau.mass*tau_es
                    #MET correction due to tau variation
                    corrected_met = MET_correction(events.SelMET, Tau_not_corrected, events.SelTau)
                    #MET correction due to jet variation
                    corrected_met = MET_correction(corrected_met, Jet_not_corrected, events.SelJet) 
                    events["SelMET","pt"] = corrected_met.pt
                    events["SelMET","phi"] = corrected_met.phi

        else:
            #if mode is Data apply jet correction
            #MET correction due to jet variation
            corrected_met = MET_correction(events.SelMET, events.SelJet, corrected_jets) 
            events["SelJet","pt"] = corrected_jets.pt
            events["SelJet","mass"] = corrected_jets.mass
            events["SelMET","pt"] = corrected_met.pt
            events["SelMET","phi"] = corrected_met.phi

        #apply cut on Tau pt using corrected energy
        events['SelTau'] = events.SelTau[events.SelTau.pt > self.cut_tau_pt]
        events['SelElectron'] = events.SelElectron[events.SelElectron.pt > self.cut_e_pt]

        return events
    
