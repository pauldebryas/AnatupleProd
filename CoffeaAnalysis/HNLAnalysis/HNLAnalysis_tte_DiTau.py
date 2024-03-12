import numpy as np
import awkward as ak
from coffea import processor

from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_common, save_anatuple_lepton, save_anatuple_tau, save_bjets, save_Event
from CoffeaAnalysis.HNLAnalysis.correction_helpers import compute_sf_e, compute_sf_tau, get_trigger_correction_tau, compute_sf_L1PreFiring, get_pileup_correction
from CoffeaAnalysis.HNLAnalysis.helpers import IsoElectron_mask, FinalTau_sel, delta_r, bjet_candidates
from CoffeaAnalysis.HNLAnalysis.HNLProcessor import HNLProcessor

class HNLAnalysis_tte(processor.ProcessorABC, HNLProcessor):
    def __init__(self, stitched_list, tag, xsecs, periods):
        HNLProcessor.__init__(self, stitched_list, tag, xsecs, periods)
        acc_dict = {}
        self.selections = self.get_selections()
        for selection in self.selections:
            acc_dict[f'n_ev_{selection}'] = processor.defaultdict_accumulator(int)
            acc_dict[f'sumw_{selection}'] = processor.defaultdict_accumulator(float)
        self._accumulator = processor.dict_accumulator(acc_dict)

        #the corresponding data sample for tte channel (HLT=DoubleTau) 
        self.dataHLT = 'Tau'

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
        events, out = self.init_process(out, events)

        # cut specific for that channel 
        self.cut_mu_iso = 0.15 # veto events with tight mu isolation for ortogonality in signal region for channels with muon
        self.cut_tau_idVSe = 6 # require tight isolation against electron for channel with reco electron

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
                    Treename = 'Events_GenuineTauES_'+Tau_ES_corr+'_'+val_corr
                    print('TAU ES corrections: saving '+ Treename)
                    events_corr = self.Lepton_selection(events, Treename)
                    events_corr, Sel_Electron, Sel_Tau1, Sel_Tau2 = self.analyse_tte(events_corr)
                    save_file, lst = self.save_anatuple_tte(events_corr, Sel_Electron, Sel_Tau1, Sel_Tau2, self.tag, save_weightcorr=False)
                    save_Event(save_file, lst, Treename)

            # Compute Tau_ES for genuineElectron
            for Tau_ES_corr in Tau_ES_corr_list:
                for val_corr in ['up','down']:
                    Treename = 'Events_GenuineElectronES_'+Tau_ES_corr+'_'+val_corr
                    print('TAU ES corrections: saving '+ Treename)
                    events_corr = self.Lepton_selection(events, Treename)
                    events_corr, Sel_Electron, Sel_Tau1, Sel_Tau2 = self.analyse_tte(events_corr)
                    save_file, lst = self.save_anatuple_tte(events_corr, Sel_Electron, Sel_Tau1, Sel_Tau2, self.tag, save_weightcorr=False)
                    save_Event(save_file, lst, Treename)

            # Compute Tau_ES for genuineMuon
            for val_corr in ['up','down']:
                Treename = 'Events_GenuineMuonES_'+val_corr
                print('TAU ES corrections: saving '+ Treename)
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
        
        # veto events with more than one electron with pfRelIso03_all <= 0.15
        events_tte = events_tte[ IsoElectron_mask(events_tte, 1, iso_cut=0.15)]

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

        # events should pass most efficient HLT (DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg)
        events_tte = events_tte[events_tte.HLT.DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg]

        if out != None:
            out[f'sumw_HLT'][self.ds] += ak.sum(events_tte.genWeight)
            out[f'n_ev_HLT'][self.ds] += len(events_tte) 

        #find electron with minimum pfRelIso03_all, in case same isolation, we take the first one by default
        Sel_Electron = events_tte.SelElectron
        
        #select the one with min isolation
        Sel_Electron = Sel_Electron[ak.min(Sel_Electron.pfRelIso03_all, axis=-1) == Sel_Electron.pfRelIso03_all]
        Sel_Electron = Sel_Electron[:,0]

        if out != None:
            out[f'sumw_eSelection'][self.ds] += ak.sum(events_tte.genWeight)
            out[f'n_ev_eSelection'][self.ds] += len(events_tte) 

        delta_r_cut = 0.5

        # select Tau1 candidate with dr(electron,Tau)>0.5 and with tau_pt >= 40. and tau_pt <= 2.1 (trigger requirement)
        self.cut_tau_pt = 40.
        self.cut_tau_eta = 2.1 
        Tau1_candidate = events_tte.SelTau[(delta_r(Sel_Electron,events_tte.SelTau)> delta_r_cut) & (events_tte.SelTau.ptcorr >= self.cut_tau_pt) & (events_tte.SelTau.eta >= self.cut_tau_eta)]

        #make sure at least 1 satisfy this condition
        events_tte = events_tte[ak.num(Tau1_candidate) >= 1]
        Sel_Electron = Sel_Electron[ak.num(Tau1_candidate) >= 1]
        Tau1_candidate = Tau1_candidate[ak.num(Tau1_candidate) >= 1]

        #Tau1 is the one with the highest isolation VSJet
        Sel_Tau1 = Tau1_candidate[ak.max(Tau1_candidate.rawDeepTau2018v2p5VSjet, axis=-1) == Tau1_candidate.rawDeepTau2018v2p5VSjet]
        Sel_Tau1 = Sel_Tau1[:,0]

        if out != None:
            out[f'sumw_drTau_e'][self.ds] += ak.sum(events_tte.genWeight)
            out[f'n_ev_drTau_e'][self.ds] += len(events_tte)

        # select Tau2 candidate with dr(e,Tau)>0.5 and dr(Tau1,Tau2)>0.5  and highest isolation VSJet
        events_tte, Sel_Electron, Sel_Tau1, Sel_Tau2 = FinalTau_sel(events_tte, Sel_Electron, Sel_Tau1, self.DeepTauVersion)

        if out != None:
            out[f'sumw_drTau2Leptons'][self.ds] += ak.sum(events_tte.genWeight)
            out[f'n_ev_drTau2Leptons'][self.ds] += len(events_tte)
        
        # !!!! must change tau1 and tau2 selection, requiring matching with trigger object beforhand
        
        # Save bjets candidates
        bjet_candidates(events_tte, Sel_Electron, Sel_Tau1, Sel_Tau2, self.period)

        if len(events_tte) == 0:
            print('0 events pass selection')
            return
        
        events_tte['matchingHLTDoubleTau'] = self.matching_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_old(events_tte, Sel_Tau1, Sel_Tau2)

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
        Trigger_eff_corr_tau1 = get_trigger_correction_tau(Sel_Tau1, events, 'Tau1', self.period) 
        Trigger_eff_corr_tau2 = get_trigger_correction_tau(Sel_Tau2, events, 'Tau2', self.period) 
        sf_L1PreFiring = compute_sf_L1PreFiring(events)
        PU_corr, PU_corr_up, PU_corr_down = get_pileup_correction(events, self.period)
        events.genWeight = events.genWeight * sf_tau1 * sf_tau2 * sf_e * Trigger_eff_corr_tau1 * Trigger_eff_corr_tau2 * sf_L1PreFiring * PU_corr
        return events
    
    def matching_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_old(self, events, Sel_Tau1, Sel_Tau2, min_dr_cut=0.2):

        # select reco tau that match HLT
        # Trigger tau is a tau (id == 15),abs(eta) < 2.2 and pt > 0 (HLT) with  TrigObj_filterBits for Tau: 2 = MediumChargedIso, 16 = HPS, 64 = di-tau
        Trigger_Tau = events.TrigObj[ (abs(events.TrigObj.id) == 15) & (abs(events.TrigObj.eta) < 2.2) & ((events.TrigObj.filterBits & (2+16+64)) != 0)] 

        cut_dr_mask_tau1 = delta_r(Sel_Tau1, Trigger_Tau) < min_dr_cut # remove too high dr matching
        cut_dr_mask_tau2 = delta_r(Sel_Tau2, Trigger_Tau) < min_dr_cut # remove too high dr matching
    
        return (ak.num(cut_dr_mask_tau1) >= 1) & (ak.num(cut_dr_mask_tau2) >= 1) 

    def save_anatuple_tte(self, events, Sel_Electron, Sel_Tau1, Sel_Tau2, tag, save_weightcorr=True):

        exclude_list = ['genPartIdx']
        
        save_file, lst = save_anatuple_common(self.ds, events, tag, 'tte', save_weightcorr)

        #info specific to the channel
        lst["nAdditionalElectron"] = np.array(events.nAdditionalElectron)
        lst["matchingHLTDoubleTau"] = np.array(events.matchingHLTDoubleTau)
        lst['channelIndex'] = np.ones(len(events))*331

        if self.mode == 'signal':
            lst['HNLmass'] = np.ones(len(events))*int(self.ds[self.ds.rfind("-") + 1:])

        lst = save_anatuple_lepton(Sel_Electron, lst, exclude_list, 'Electron')
        lst = save_anatuple_tau(events, Sel_Tau1, lst, exclude_list, self.mode, 'Tau1')
        lst = save_anatuple_tau(events, Sel_Tau2, lst, exclude_list, self.mode, 'Tau2')

        return save_file, lst

    def postprocess(self, accumulator):
        return accumulator