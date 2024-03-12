import numpy as np
import awkward as ak
from coffea import processor

from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_common, save_anatuple_lepton, save_anatuple_tau, save_bjets, save_Event
from CoffeaAnalysis.HNLAnalysis.correction_helpers import compute_sf_tau, compute_sf_mu, compute_sf_e, get_trigger_correction_mu, compute_sf_L1PreFiring, get_pileup_correction
from CoffeaAnalysis.HNLAnalysis.helpers import IsoMuon_mask, IsoElectron_mask, Trigger_Muon_sel, FinalTau_sel, delta_r, bjet_candidates
from CoffeaAnalysis.HNLAnalysis.HNLProcessor import HNLProcessor

class HNLAnalysis_tem(processor.ProcessorABC, HNLProcessor):
    def __init__(self, stitched_list, tag, xsecs, periods):
        HNLProcessor.__init__(self, stitched_list, tag, xsecs, periods)
        acc_dict = {}
        self.selections = self.get_selections()
        for selection in self.selections:
            acc_dict[f'n_ev_{selection}'] = processor.defaultdict_accumulator(int)
            acc_dict[f'sumw_{selection}'] = processor.defaultdict_accumulator(float)
        self._accumulator = processor.dict_accumulator(acc_dict)

        #the corresponding data sample for tem channel (HLT=IsoMu24) 
        self.dataHLT = 'SingleMuon'

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
            'NoAdditionalElectron',
            'HLT',
            'MuTriggerMatching',
            'eSelection',
            'drTauLeptons',
            'corrections'
        ]

    # we will receive a NanoEvents
    def process(self, events):

        out = self.accumulator.identity()
        events, out = self.init_process(out, events)

        # cut specific for that channel 
        self.cut_tau_idVSe = 6 # require tight isolation against electron for channel with reco electron

        print('Running main analysis')
        # Do the general lepton selection
        events_tem = self.Lepton_selection(events)

        # Apply the cuts and select leptons
        events_tem, Sel_Muon, Sel_Electron, Sel_Tau = self.analyse_tem(events_tem, out)

        # Save anatuple
        save_file, lst = self.save_anatuple_tem(events_tem, Sel_Muon, Sel_Electron, Sel_Tau, self.tag, save_weightcorr=True)
        save_Event(save_file, lst, 'Events')
        if self.mode != 'Data':
            save_bjets(save_file, events_tem) #for bTagSF

        if self.mode != 'Data':
            Tau_ES_corr_list = ['DM0', 'DM1', '3prong']
            # Compute Tau_ES for genuineTau
            for Tau_ES_corr in Tau_ES_corr_list:
                for val_corr in ['up','down']:
                    Treename = 'Events_GenuineTauES_'+Tau_ES_corr+'_'+val_corr
                    print('TAU ES corrections: saving '+ Treename)
                    events_corr = self.Lepton_selection(events, Treename)
                    events_corr, Sel_Muon, Sel_Electron, Sel_Tau = self.analyse_tem(events_corr)
                    save_file, lst = self.save_anatuple_tem(events_corr, Sel_Muon, Sel_Electron, Sel_Tau, self.tag, save_weightcorr=False)
                    save_Event(save_file, lst, Treename)

            # Compute Tau_ES for genuineElectron
            for Tau_ES_corr in Tau_ES_corr_list:
                for val_corr in ['up','down']:
                    Treename = 'Events_GenuineElectronES_'+Tau_ES_corr+'_'+val_corr
                    print('TAU ES corrections: saving '+ Treename)
                    events_corr = self.Lepton_selection(events, Treename)
                    events_corr, Sel_Muon, Sel_Electron, Sel_Tau = self.analyse_tem(events_corr)
                    save_file, lst = self.save_anatuple_tem(events_corr, Sel_Muon, Sel_Electron, Sel_Tau, self.tag, save_weightcorr=False)
                    save_Event(save_file, lst, Treename)

            # Compute Tau_ES for genuineMuon
            for val_corr in ['up','down']:
                Treename = 'Events_GenuineMuonES_'+val_corr
                print('TAU ES corrections: saving '+ Treename)
                events_corr = self.Lepton_selection(events, Treename)
                events_corr, Sel_Muon, Sel_Electron, Sel_Tau = self.analyse_tem(events_corr)
                save_file, lst = self.save_anatuple_tem(events_corr, Sel_Muon, Sel_Electron, Sel_Tau, self.tag, save_weightcorr=False)
                save_Event(save_file, lst, Treename)

        return out

    def analyse_tem(self, events, out= None):
        # If out is None, do note save the cutflow

        # select tem events: require at least 1 reco mu, 1 reco tau and 1 reco e
        events_tem = events[(ak.num(events.SelElectron) >= 1) & (ak.num(events.SelMuon) >= 1) & (ak.num(events.SelTau) >= 1)]

        if out != None:
            out[f'sumw_AtLeast1tau1mu1e'][self.ds] += ak.sum(events_tem.genWeight)
            out[f'n_ev_AtLeast1tau1mu1e'][self.ds] += len(events_tem)

        # veto events with more than one muon with pfRelIso03_all <= 0.15
        events_tem = events_tem[IsoMuon_mask(events_tem, 1, iso_cut=0.15)]

        if out != None:
            out[f'sumw_NoAdditionalIsoMuon'][self.ds] += ak.sum(events_tem.genWeight)
            out[f'n_ev_NoAdditionalIsoMuon'][self.ds] += len(events_tem)

        # veto events with more than one electron with pfRelIso03_all <= 0.15
        events_tem = events_tem[ IsoElectron_mask(events_tem, 1, iso_cut=0.15)]

        if out != None:
            out[f'sumw_NoAdditionalElectron'][self.ds] += ak.sum(events_tem.genWeight)
            out[f'n_ev_NoAdditionalElectron'][self.ds] += len(events_tem)

        # save info if an extra muon exist (with 0.15 < pfRelIso03_all < 0.4)
        events_tem['nAdditionalMuon'] = ak.num(events_tem.SelMuon[events_tem.SelMuon.pfRelIso03_all > 0.15])

        # events should pass most efficient HLT (for now only 1: IsoMu)
        if self.period == '2018':
            events_tem = events_tem[events_tem.HLT.IsoMu24]
        if self.period == '2017':
            events_tem = events_tem[events_tem.HLT.IsoMu27]
        if self.period == '2016':
            events_tem = events_tem[events_tem.HLT.IsoMu24]

        if out != None:
            out[f'sumw_HLT'][self.ds] += ak.sum(events_tem.genWeight)
            out[f'n_ev_HLT'][self.ds] += len(events_tem)

        events_tem, Sel_Muon = Trigger_Muon_sel(events_tem, self.period)

        if out != None:
            out[f'sumw_MuTriggerMatching'][self.ds] += ak.sum(events_tem.genWeight)
            out[f'n_ev_MuTriggerMatching'][self.ds] += len(events_tem)

        #find electron with minimum pfRelIso03_all, in case same isolation, we take the first one by default
        Sel_Electron = events_tem.SelElectron
        
        #select the one with min isolation
        Sel_Electron = Sel_Electron[ak.min(Sel_Electron.pfRelIso03_all, axis=-1) == Sel_Electron.pfRelIso03_all]
        Sel_Electron = Sel_Electron[:,0]

        if out != None:
            out[f'sumw_eSelection'][self.ds] += ak.sum(events_tem.genWeight)
            out[f'n_ev_eSelection'][self.ds] += len(events_tem)

        # select Tau candidate with dr(muon,Tau)>0.5 and dr(electron,Tau)>0.5  and highest isolation VSJet
        events_tem, Sel_Electron, Sel_Muon, Sel_Tau = FinalTau_sel(events_tem, Sel_Electron, Sel_Muon, self.DeepTauVersion)

        if out != None:
            out[f'sumw_drTauLeptons'][self.ds] += ak.sum(events_tem.genWeight)
            out[f'n_ev_drTauLeptons'][self.ds] += len(events_tem)

        # Save bjets candidates
        bjet_candidates(events_tem, Sel_Electron, Sel_Muon, Sel_Tau, self.period)

        if len(events_tem) == 0:
            print('0 events pass selection')
            return

        # Apply corrections for MC
        if self.mode != 'Data':
            events_tem = self.compute_corrections_tem(events_tem, Sel_Muon, Sel_Electron, Sel_Tau)

        if out != None:
            out[f'sumw_corrections'][self.ds] += ak.sum(events_tem.genWeight)
            out[f'n_ev_corrections'][self.ds] += len(events_tem)

        return events_tem, Sel_Muon, Sel_Electron, Sel_Tau

    def compute_corrections_tem(self, events, Sel_Muon, Sel_Electron, Sel_Tau):
        #computing sf corrections       
        sf_tau = compute_sf_tau(Sel_Tau, events, 'Tau', self.period, self.DeepTauVersion)
        sf_mu = compute_sf_mu(Sel_Muon, events, 'Muon', self.period)
        sf_e = compute_sf_e(Sel_Electron, events, 'Electron', self.period)
        Trigger_eff_corr_mu = get_trigger_correction_mu(Sel_Muon, events, 'Muon', self.period) # apply Trigger sf to the Muon that match HLT
        sf_L1PreFiring = compute_sf_L1PreFiring(events)
        PU_corr, PU_corr_up, PU_corr_down = get_pileup_correction(events, self.period)
        events.genWeight = events.genWeight * sf_tau * sf_mu * sf_e * Trigger_eff_corr_mu * sf_L1PreFiring * PU_corr
        return events
    
    def save_anatuple_tem(self, events, Sel_Muon, Sel_Electron, Sel_Tau, tag, save_weightcorr=True):

        exclude_list = ['genPartIdx']
        
        save_file, lst = save_anatuple_common(self.ds, events, tag, 'tem', save_weightcorr)
        
        #info specific to the channel
        lst["nAdditionalMuon"] = np.array(events.nAdditionalMuon)
        lst['channelIndex'] = np.ones(len(events))*312

        if self.mode == 'signal':
            lst['HNLmass'] = np.ones(len(events))*int(self.ds[self.ds.rfind("-") + 1:])

        lst = save_anatuple_lepton(Sel_Muon, lst, exclude_list, 'Muon')
        lst = save_anatuple_lepton(Sel_Electron, lst, exclude_list, 'Electron')
        lst = save_anatuple_tau(events, Sel_Tau, lst, exclude_list, self.mode, 'Tau')

        return save_file, lst

    def postprocess(self, accumulator):
        return accumulator