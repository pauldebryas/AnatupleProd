import numpy as np
import awkward as ak
from coffea import processor

from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_common, save_anatuple_lepton, save_anatuple_tau, save_bjets, save_Event
from CoffeaAnalysis.HNLAnalysis.correction_helpers import compute_sf_e, compute_sf_tau, get_trigger_correction_e, compute_sf_L1PreFiring, get_pileup_correction
from CoffeaAnalysis.HNLAnalysis.helpers import IsoElectron_mask, Ele32_Electron_sel, FinalTau_sel, delta_r, bjet_candidates
from CoffeaAnalysis.HNLAnalysis.HNLProcessor import HNLProcessor

class HNLAnalysis_tee(processor.ProcessorABC, HNLProcessor):
    def __init__(self, stitched_list, tag, xsecs, periods):
        HNLProcessor.__init__(self, stitched_list, tag, xsecs, periods)
        acc_dict = {}
        self.selections = self.get_selections()
        for selection in self.selections:
            acc_dict[f'n_ev_{selection}'] = processor.defaultdict_accumulator(int)
            acc_dict[f'sumw_{selection}'] = processor.defaultdict_accumulator(float)
        self._accumulator = processor.dict_accumulator(acc_dict)

        #the corresponding data sample for tee channel (HLT=Ele32)
        self.dataHLT = 'EGamma'

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
            'corrections']

    # we will receive a NanoEvents
    def process(self, events):

        out = self.accumulator.identity()
        events, out = self.init_process(out, events)

        # cut specific for that channel 
        self.cut_mu_iso = 0.15 # veto events with tight mu isolation for ortogonality in signal region for channels with muon
        self.cut_tau_idVSe = 6 # require tight isolation against electron for channel with reco electron

        # Do the general lepton selection
        events_tee = self.Lepton_selection(events)

        # Apply the cuts and select leptons
        events_tee, Sel_Electron, Sel_Electron2, Sel_Tau = self.analyse_tee(events_tee, out)

        # Save anatuple
        save_file, lst = self.save_anatuple_tee(events_tee, Sel_Electron, Sel_Electron2, Sel_Tau, self.tag, save_weightcorr=True)
        save_Event(save_file, lst, 'Events')
        if self.mode != 'Data':
            save_bjets(save_file, events_tee) #for bTagSF

        if self.mode != 'Data':
            Tau_ES_corr_list = ['DM0', 'DM1', '3prong']
            # Compute Tau_ES for genuineTau
            for Tau_ES_corr in Tau_ES_corr_list:
                for val_corr in ['up','down']:
                    Treename = 'Events_GenuineTauES_'+Tau_ES_corr+'_'+val_corr
                    print('TAU ES corrections: saving '+ Treename)
                    events_corr = self.Lepton_selection(events, Treename)
                    events_corr, Sel_Electron, Sel_Electron2, Sel_Tau = self.analyse_tee(events_corr)
                    save_file, lst = self.save_anatuple_tee(events_corr, Sel_Electron, Sel_Electron2, Sel_Tau, self.tag, save_weightcorr=False)
                    save_Event(save_file, lst, Treename)

            # Compute Tau_ES for genuineElectron
            for Tau_ES_corr in Tau_ES_corr_list:
                for val_corr in ['up','down']:
                    Treename = 'Events_GenuineElectronES_'+Tau_ES_corr+'_'+val_corr
                    print('TAU ES corrections: saving '+ Treename)
                    events_corr = self.Lepton_selection(events, Treename)
                    events_corr, Sel_Electron, Sel_Electron2, Sel_Tau = self.analyse_tee(events_corr)
                    save_file, lst = self.save_anatuple_tee(events_corr, Sel_Electron, Sel_Electron2, Sel_Tau, self.tag, save_weightcorr=False)
                    save_Event(save_file, lst, Treename)

            # Compute Tau_ES for genuineMuon
            for val_corr in ['up','down']:
                Treename = 'Events_GenuineMuonES_'+val_corr
                print('TAU ES corrections: saving '+ Treename)
                events_corr = self.Lepton_selection(events, Treename)
                events_corr, Sel_Electron, Sel_Electron2, Sel_Tau = self.analyse_tee(events_corr)
                save_file, lst = self.save_anatuple_tee(events_corr, Sel_Electron, Sel_Electron2, Sel_Tau, self.tag, save_weightcorr=False)
                save_Event(save_file, lst, Treename)
        return out

    def analyse_tee(self, events, out= None):
        # If out is None, do note save the cutflow

        # select tee events: require 2 reco e and 1 reco tau 
        events_tee = events[(ak.num(events.SelElectron) >= 2) & (ak.num(events.SelTau) >= 1)]

        if out != None:
            out[f'sumw_AtLeast1tau2e'][self.ds] += ak.sum(events_tee.genWeight)
            out[f'n_ev_AtLeast1tau2e'][self.ds] += len(events_tee)

        # veto events with more than two electrons with pfRelIso03_all <= 0.15
        events_tee = events_tee[ IsoElectron_mask(events_tee, 2, iso_cut=0.15)]

        if out != None:
            out[f'sumw_NoAdditionalIsoElectron'][self.ds] += ak.sum(events_tee.genWeight)
            out[f'n_ev_NoAdditionalIsoElectron'][self.ds] += len(events_tee)

        # save info if an extra electron exist (with 0.15 <pfRelIso03_all < 0.4)
        events_tee['nAdditionalElectron'] = ak.num(events_tee.SelElectron[events_tee.SelMuon.pfRelIso03_all > 0.15])

        # remove events with more than one isolated muon (assure orthogonality with ttm tmm tem)
        events_tee = events_tee[ak.num(events_tee.SelMuon) == 0]

        if out != None:
            out[f'sumw_noIsoMuon'][self.ds] += ak.sum(events_tee.genWeight)
            out[f'n_ev_noIsoMuon'][self.ds] += len(events_tee)

        # events should pass most efficient HLT (Ele32_WPTight_Gsf_L1DoubleEG)
        events_tee = events_tee[events_tee.HLT.Ele32_WPTight_Gsf_L1DoubleEG]

        if out != None:
            out[f'sumw_HLT'][self.ds] += ak.sum(events_tee.genWeight)
            out[f'n_ev_HLT'][self.ds] += len(events_tee)

        #find electron with minimum pfRelIso03_all, and matching Ele32 trigger. In case same isolation, we take the first one by default
        events_tee, Sel_Electron = Ele32_Electron_sel(events_tee)

        if out != None:
            out[f'sumw_eSelection'][self.ds] += ak.sum(events_tee.genWeight)
            out[f'n_ev_eSelection'][self.ds] += len(events_tee) 

        delta_r_cut = 0.5

        # select Electron2, that should be away from first Electron: dr(e2,e)>0.5
        e2_candidate = events_tee.SelElectron[(delta_r(Sel_Electron,events_tee.SelElectron)> delta_r_cut)]

        #make sure at least 1 satisfy this condition
        events_tee = events_tee[ak.num(e2_candidate) >= 1]
        Sel_Electron = Sel_Electron[ak.num(e2_candidate) >= 1]
        e2_candidate = e2_candidate[ak.num(e2_candidate) >= 1]

        #e2 is the one with the minimum pfRelIso03_all
        Sel_Electron2 = e2_candidate[ak.min(e2_candidate.pfRelIso03_all, axis=-1) == e2_candidate.pfRelIso03_all]
        Sel_Electron2 = Sel_Electron2[:,0]

        if out != None:
            out[f'sumw_e2sel'][self.ds] += ak.sum(events_tee.genWeight)
            out[f'n_ev_e2sel'][self.ds] += len(events_tee)

        # select Tau candidate with dr(e1,Tau)>0.5 and dr(e2,Tau)>0.5 and highest isolation VSJet
        events_tee, Sel_Electron, Sel_Electron2, Sel_Tau = FinalTau_sel(events_tee, Sel_Electron, Sel_Electron2, self.DeepTauVersion)

        if out != None:
            out[f'sumw_tausel'][self.ds] += ak.sum(events_tee.genWeight)
            out[f'n_ev_tausel'][self.ds] += len(events_tee)

        # Save bjets candidates
        bjet_candidates(events_tee, Sel_Electron, Sel_Electron2, Sel_Tau, self.period)

        if len(events_tee) == 0:
            print('0 events pass selection')
            return

        # Apply corrections for MC
        if self.mode != 'Data':
            events_tee = self.compute_corrections_tee(events_tee, Sel_Electron, Sel_Electron2, Sel_Tau)

        if out != None:
            out[f'sumw_corrections'][self.ds] += ak.sum(events_tee.genWeight)
            out[f'n_ev_corrections'][self.ds] += len(events_tee)
        
        return events_tee, Sel_Electron, Sel_Electron2, Sel_Tau

    def compute_corrections_tee(self, events, Sel_Electron, Sel_Electron2, Sel_Tau):
        #computing sf corrections        
        sf_tau = compute_sf_tau(Sel_Tau, events, 'Tau', self.period, self.DeepTauVersion)
        sf_e1 = compute_sf_e(Sel_Electron, events, 'Electron1', self.period)
        sf_e2 = compute_sf_e(Sel_Electron2, events, 'Electron2', self.period)
        Trigger_eff_corr_e1 = get_trigger_correction_e(Sel_Electron, events, 'Electron1', self.period) # apply Trigger sf to the Electron that match HLT
        Trigger_eff_corr_e2 = get_trigger_correction_e(Sel_Electron2, events, 'Electron2', self.period) # compute Trigger sf for Electron2
        sf_L1PreFiring = compute_sf_L1PreFiring(events)
        PU_corr, PU_corr_up, PU_corr_down = get_pileup_correction(events, self.period)
        events.genWeight = events.genWeight * sf_tau * sf_e1 * sf_e2 * Trigger_eff_corr_e1 * sf_L1PreFiring * PU_corr
        return events
    
    def save_anatuple_tee(self, events, Sel_Electron, Sel_Electron2, Sel_Tau, tag, save_weightcorr=True):

        exclude_list = ['genPartIdx']
        
        save_file, lst = save_anatuple_common(self.ds, events, tag, 'tee', save_weightcorr)

        #info specific to the channel
        lst["nAdditionalElectron"] = np.array(events.nAdditionalElectron)
        lst['channelIndex'] = np.ones(len(events))*311

        if self.mode == 'signal':
            lst['HNLmass'] = np.ones(len(events))*int(self.ds[self.ds.rfind("-") + 1:])
        
        #order electron by pt
        mask_ptmax = Sel_Electron.pt >= Sel_Electron2.pt
        Sel_Electron_ptmax = ak.where(mask_ptmax, Sel_Electron, Sel_Electron2)
        Sel_Electron_ptmin = ak.where(~mask_ptmax, Sel_Electron, Sel_Electron2)

        lst = save_anatuple_lepton(Sel_Electron_ptmax, lst, exclude_list, 'Electron1')
        lst = save_anatuple_lepton(Sel_Electron_ptmin, lst, exclude_list, 'Electron2')
        lst = save_anatuple_tau(events, Sel_Tau, lst, exclude_list, self.mode, 'Tau')

        return save_file, lst
    
    def postprocess(self, accumulator):
        return accumulator
