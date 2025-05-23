import numpy as np
import awkward as ak
from coffea import processor
from collections import defaultdict
import copy

from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_common, save_anatuple_lepton, save_anatuple_tau, save_bjets, save_Event, add_gen_matching_info
from CoffeaAnalysis.HNLAnalysis.correction_helpers import compute_sf_e, compute_sf_tau, get_trigger_correction_e, compute_sf_L1PreFiring, get_pileup_correction, get_BTag_sf
from CoffeaAnalysis.HNLAnalysis.helpers import IsoElectron_mask, Trigger_Electron_sel, FinalTau_sel, delta_r, bjet_candidates
from CoffeaAnalysis.HNLAnalysis.HNLProcessor import HNLProcessor

# test with Ele32_WPTight_Gsf_L1DoubleEG trigger
class HNLAnalysis_tte(processor.ProcessorABC, HNLProcessor):
    def __init__(self, stitched_list, tag, xsecs, periods, dataHLT, debugMode, sample_name):
        HNLProcessor.__init__(self, stitched_list, tag, xsecs, periods, dataHLT, debugMode)
        self.acc_dict = {}
        self.selections = self.get_selections()
        for selection in self.selections:
            self.acc_dict[f'n_ev_{selection}'] = defaultdict(int)
            self.acc_dict[f'sumw_{selection}'] = defaultdict(int)
        self._accumulator = self.acc_dict
        self.sample_name = sample_name

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

        out = copy.deepcopy(self._accumulator)
        events, out = self.init_process(out, events)

        # cut specific for that channel 
        #self.cut_mu_iso = 0.15 # veto events with tight mu isolation for ortogonality in signal region for channels with muon
        self.cut_tau_idVSe = 6 # require tight isolation against electron for channel with reco electron

        print('Running main analysis')
        # Do the general lepton selection
        events_tte = self.Lepton_selection(events)

        # Apply the cuts and select leptons
        events_tte, Sel_Electron, Sel_Tau1, Sel_Tau2 = self.analyse_tte(events_tte, out)

        # Save anatuple
        save_file, lst = self.save_anatuple_tte(events_tte, Sel_Electron, Sel_Tau1, Sel_Tau2, self.tag, save_weightcorr=True)
        save_Event(save_file, lst, 'Events')
        if self.mode != 'Data':
            save_bjets(save_file, events_tte) #for bTagSF

        if self.mode != 'Data':
            Tau_ES_corr_list = ['DM0', 'DM1', '3prong']
            # Compute Tau_ES for genuineTau
            for Tau_ES_corr in Tau_ES_corr_list:
                for val_corr in ['up','down']:
                    Treename = 'Events_GenuineTauES_'+Tau_ES_corr+'_'+self.period+'_'+val_corr
                    print('TAU ES corrections: saving '+ Treename)
                    events_corr = self.Lepton_selection(events, Treename)
                    events_corr, Sel_Electron, Sel_Tau1, Sel_Tau2 = self.analyse_tte(events_corr)
                    save_file, lst = self.save_anatuple_tte(events_corr, Sel_Electron, Sel_Tau1, Sel_Tau2, self.tag, save_weightcorr=False)
                    save_Event(save_file, lst, Treename)

            # Compute Tau_ES for genuineElectron
            for Tau_ES_corr in Tau_ES_corr_list:
                for val_corr in ['up','down']:
                    Treename = 'Events_GenuineElectronES_'+Tau_ES_corr+'_'+self.period+'_'+val_corr
                    print('TAU ES corrections: saving '+ Treename)
                    events_corr = self.Lepton_selection(events, Treename)
                    events_corr, Sel_Electron, Sel_Tau1, Sel_Tau2 = self.analyse_tte(events_corr)
                    save_file, lst = self.save_anatuple_tte(events_corr, Sel_Electron, Sel_Tau1, Sel_Tau2, self.tag, save_weightcorr=False)
                    save_Event(save_file, lst, Treename)

            # Compute Tau_ES for genuineMuon
            for val_corr in ['up','down']:
                Treename = 'Events_GenuineMuonES'+'_'+self.period+'_'+val_corr
                print('TAU ES corrections: saving '+ Treename)
                events_corr = self.Lepton_selection(events, Treename)
                events_corr, Sel_Electron, Sel_Tau1, Sel_Tau2 = self.analyse_tte(events_corr)
                save_file, lst = self.save_anatuple_tte(events_corr, Sel_Electron, Sel_Tau1, Sel_Tau2, self.tag, save_weightcorr=False)
                save_Event(save_file, lst, Treename)

            # Compute Electron_ES: syst (correlated between the years), gain (uncorrelated between the years)
            #for electron_ES_corr in ['syst', 'gain']:
            for val_corr in ['up','down']:
                Treename = 'Events_ElectronES'+'_'+self.period+'_'+val_corr
                print('ELE ES corrections: saving '+ Treename)
                events_corr = self.Lepton_selection(events, Treename)
                events_corr, Sel_Electron, Sel_Tau1, Sel_Tau2 = self.analyse_tte(events_corr)
                save_file, lst = self.save_anatuple_tte(events_corr, Sel_Electron, Sel_Tau1, Sel_Tau2, self.tag, save_weightcorr=False)
                save_Event(save_file, lst, Treename)

            # Compute Electron_ER (correlated between the years)
            for val_corr in ['up','down']:
                Treename = 'Events_ElectronER_'+val_corr
                print('ELE ER corrections: saving '+ Treename)
                events_corr = self.Lepton_selection(events, Treename)
                events_corr, Sel_Electron, Sel_Tau1, Sel_Tau2 = self.analyse_tte(events_corr)
                save_file, lst = self.save_anatuple_tte(events_corr, Sel_Electron, Sel_Tau1, Sel_Tau2, self.tag, save_weightcorr=False)
                save_Event(save_file, lst, Treename)

        # JET corrections
        Jet_corr_list = ['JES', 'JER']
        for Jet_corr in Jet_corr_list:
            for val_corr in ['up','down']:
                Treename = 'Events_'+Jet_corr+'_'+self.period+'_'+val_corr
                print('JET corrections: saving '+ Treename)
                events_corr = self.Lepton_selection(events, Treename)
                events_corr, Sel_Electron, Sel_Tau1, Sel_Tau2 = self.analyse_tte(events_corr)
                save_file, lst = self.save_anatuple_tte(events_corr, Sel_Electron, Sel_Tau1, Sel_Tau2, self.tag, save_weightcorr=False)
                save_Event(save_file, lst, Treename)

        return out

    def analyse_tte(self, events, out= None):
        # If out is None, do note save the cutflow

        # select tte events: require at least 2 reco tau and 1 reco e 
        events_tte = events[(ak.num(events.SelElectron) >= 1) & (ak.num(events.SelTau) >= 2)]

        if out != None:
            out[f'sumw_AtLeast2tau1e'][self.ds] += ak.sum(events_tte.genWeight)
            out[f'n_ev_AtLeast2tau1e'][self.ds] += len(events_tte)
        
        # veto events with more than one electron with pfRelIso03_all <= 0.4
        events_tte = events_tte[ IsoElectron_mask(events_tte, 1, iso_cut=0.4)]

        if out != None:
            out[f'sumw_NoAdditionalIsoElectron'][self.ds] += ak.sum(events_tte.genWeight)
            out[f'n_ev_NoAdditionalIsoElectron'][self.ds] += len(events_tte)

        # save info if an extra electron exist (with 0.15 < pfRelIso03_all < 0.4 )
        events_tte['nAdditionalElectron'] = ak.num(events_tte.SelElectron[events_tte.SelElectron.pfRelIso03_all > 0.15])

        # remove events with more than one isolated muon (assure orthogonality with ttm tmm tem)
        events_tte = events_tte[ak.num(events_tte.SelMuon) == 0]

        if out != None:
            out[f'sumw_noIsoMuon'][self.ds] += ak.sum(events_tte.genWeight)
            out[f'n_ev_noIsoMuon'][self.ds] += len(events_tte)

        # events should pass most efficient HLT (Ele32_WPTight_Gsf_L1DoubleEG)
        if self.period == '2018':
            events_tte = events_tte[events_tte.HLT.Ele32_WPTight_Gsf]
        if self.period == '2017':
            events_tte = events_tte[events_tte.HLT.Ele32_WPTight_Gsf_L1DoubleEG]
        if self.period == '2016':
            events_tte = events_tte[events_tte.HLT.Ele25_eta2p1_WPTight_Gsf]
        if self.period == '2016_HIPM':
            events_tte = events_tte[events_tte.HLT.Ele25_eta2p1_WPTight_Gsf]

        if out != None:
            out[f'sumw_HLT'][self.ds] += ak.sum(events_tte.genWeight)
            out[f'n_ev_HLT'][self.ds] += len(events_tte) 

        #find electron with minimum pfRelIso03_all, and matching Ele32 trigger. In case same isolation, we take the first one by default
        events_tte, Sel_Electron = Trigger_Electron_sel(events_tte, self.period)

        if out != None:
            out[f'sumw_eSelection'][self.ds] += ak.sum(events_tte.genWeight)
            out[f'n_ev_eSelection'][self.ds] += len(events_tte) 

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

        if out != None:
            out[f'sumw_drTau_e'][self.ds] += ak.sum(events_tte.genWeight)
            out[f'n_ev_drTau_e'][self.ds] += len(events_tte)

        # select Tau2 candidate with dr(e,Tau)>0.5 and dr(Tau1,Tau2)>0.5  and highest isolation VSJet
        events_tte, Sel_Electron, Sel_Tau1, Sel_Tau2 = FinalTau_sel(events_tte, Sel_Electron, Sel_Tau1, self.DeepTauVersion)

        if out != None:
            out[f'sumw_drTau2Leptons'][self.ds] += ak.sum(events_tte.genWeight)
            out[f'n_ev_drTau2Leptons'][self.ds] += len(events_tte)
        
        # Save bjets candidates
        bjet_candidates(events_tte, Sel_Electron, Sel_Tau1, Sel_Tau2, self.period)

        #if len(events_tte) == 0:
        #    print('0 events pass selection')
        #    return

        # Apply corrections for MC
        if self.mode != 'Data':
            events_tte = self.compute_corrections_tte(events_tte, Sel_Electron, Sel_Tau1, Sel_Tau2)

        if out != None:
            out[f'sumw_corrections'][self.ds] += ak.sum(events_tte.genWeight)
            out[f'n_ev_corrections'][self.ds] += len(events_tte)

        return events_tte, Sel_Electron, Sel_Tau1, Sel_Tau2

    def compute_corrections_tte(self, events, Sel_Electron, Sel_Tau1, Sel_Tau2):
        #computing sf corrections        
        sf_tau1 = compute_sf_tau(Sel_Tau1, events, 'Tau1', self.period, self.DeepTauVersion)
        sf_tau2 = compute_sf_tau(Sel_Tau2, events, 'Tau2', self.period, self.DeepTauVersion)
        sf_e = compute_sf_e(Sel_Electron, events, 'Electron', self.period)
        Trigger_eff_corr_e = get_trigger_correction_e(Sel_Electron, events, 'Electron', self.period) 
        sf_L1PreFiring = compute_sf_L1PreFiring(events)
        PU_corr, PU_corr_up, PU_corr_down = get_pileup_correction(events, self.period)
        BTag_sf = get_BTag_sf(events, self.period)
        events['genWeight'] = events.genWeight * sf_tau1 * sf_tau2 * sf_e * Trigger_eff_corr_e * sf_L1PreFiring * PU_corr * BTag_sf

        return events

    def save_anatuple_tte(self, events, Sel_Electron, Sel_Tau1, Sel_Tau2, tag, save_weightcorr=True):
        exclude_list = ['genPartIdx']
        
        save_file, lst = save_anatuple_common(self.ds, events, tag, self.period, 'tte', save_weightcorr, self.sample_name)

        #info specific to the channel
        lst["nAdditionalElectron"] = np.array(events.nAdditionalElectron)
        lst['channelIndex'] = np.ones(len(events))*331

        if self.mode == 'signal':
            lst['HNLmass'] = np.ones(len(events))*int(self.ds[self.ds.rfind("-") + 1:])
        if self.mode != 'Data':
            Sel_Electron = add_gen_matching_info(events, Sel_Electron)
            Sel_Tau1  = add_gen_matching_info(events, Sel_Tau1)
            Sel_Tau2 = add_gen_matching_info(events, Sel_Tau2)

        #order tau by pt
        mask_ptmax = Sel_Tau1.pt >= Sel_Tau2.pt
        Sel_Tau_ptmax = ak.where(mask_ptmax, Sel_Tau1, Sel_Tau2)
        Sel_Tau_ptmin = ak.where(~mask_ptmax, Sel_Tau1, Sel_Tau2)

        lst = save_anatuple_lepton(events, Sel_Electron, lst, exclude_list, 'Electron')
        lst = save_anatuple_tau(events, Sel_Tau_ptmax, lst, exclude_list, self.mode, 'Tau1')
        lst = save_anatuple_tau(events, Sel_Tau_ptmin, lst, exclude_list, self.mode, 'Tau2')

        return save_file, lst

    def postprocess(self, accumulator):
        return accumulator