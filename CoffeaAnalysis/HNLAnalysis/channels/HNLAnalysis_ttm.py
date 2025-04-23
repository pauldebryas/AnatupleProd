import numpy as np
import awkward as ak
from coffea import processor
from collections import defaultdict
import copy

from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_common, save_anatuple_lepton, save_anatuple_tau, save_bjets, save_Event, add_gen_matching_info
from CoffeaAnalysis.HNLAnalysis.correction_helpers import compute_sf_mu, compute_sf_tau, get_trigger_correction_mu, compute_sf_L1PreFiring, get_pileup_correction, get_BTag_sf
from CoffeaAnalysis.HNLAnalysis.helpers import IsoMuon_mask, Trigger_Muon_sel, FinalTau_sel, delta_r, bjet_candidates
from CoffeaAnalysis.HNLAnalysis.HNLProcessor import HNLProcessor

class HNLAnalysis_ttm(processor.ProcessorABC, HNLProcessor):
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
            'AtLeast2tau1mu',
            'NoAdditionalIsoMuon',
            'NoIsoElectron',
            'HLT',
            'MuTriggerMatching',
            'drMuTau',
            'drTau2Leptons',
            'corrections'
        ]

    # we will receive a NanoEvents
    def process(self, events):

        out = copy.deepcopy(self._accumulator)
        events, out = self.init_process(out, events)

        # cut specific for that channel
        #self.cut_e_iso = 0.15 # veto events with tight e isolation for ortogonality only in signal region fors channels with electrons

        print('Running main analysis')
        # Do the general lepton selection
        events_ttm = self.Lepton_selection(events)

        # Apply the cuts and select leptons
        events_ttm, Sel_Muon, Sel_Tau1, Sel_Tau2 = self.analyse_ttm(events_ttm, out)

        # Save anatuple
        save_file, lst = self.save_anatuple_ttm(events_ttm, Sel_Muon, Sel_Tau1, Sel_Tau2, self.tag, save_weightcorr=True)
        save_Event(save_file, lst, 'Events')

        if self.mode != 'Data':
            save_bjets(save_file, events_ttm) #for bTagSF
        if self.mode != 'Data':
            Tau_ES_corr_list = ['DM0', 'DM1', '3prong']
            # Compute Tau_ES for genuineTau
            for Tau_ES_corr in Tau_ES_corr_list:
                for val_corr in ['up','down']:
                    Treename = 'Events_GenuineTauES_'+Tau_ES_corr+'_'+self.period+'_'+val_corr
                    print('TAU ES corrections: saving '+ Treename)
                    events_corr = self.Lepton_selection(events, Treename)
                    events_corr, Sel_Muon, Sel_Tau1, Sel_Tau2 = self.analyse_ttm(events_corr)
                    save_file, lst = self.save_anatuple_ttm(events_corr, Sel_Muon, Sel_Tau1, Sel_Tau2, self.tag, save_weightcorr=False)
                    save_Event(save_file, lst, Treename)

            # Compute Tau_ES for genuineElectron
            for Tau_ES_corr in Tau_ES_corr_list:
                for val_corr in ['up','down']:
                    Treename = 'Events_GenuineElectronES_'+Tau_ES_corr+'_'+self.period+'_'+val_corr
                    print('TAU ES corrections: saving '+ Treename)
                    events_corr = self.Lepton_selection(events, Treename)
                    events_corr, Sel_Muon, Sel_Tau1, Sel_Tau2 = self.analyse_ttm(events_corr)
                    save_file, lst = self.save_anatuple_ttm(events_corr, Sel_Muon, Sel_Tau1, Sel_Tau2, self.tag, save_weightcorr=False)
                    save_Event(save_file, lst, Treename)

            # Compute Tau_ES for genuineMuon
            for val_corr in ['up','down']:
                Treename = 'Events_GenuineMuonES'+'_'+self.period+'_'+val_corr
                print('TAU ES corrections: saving '+ Treename)
                events_corr = self.Lepton_selection(events, Treename)
                events_corr, Sel_Muon, Sel_Tau1, Sel_Tau2 = self.analyse_ttm(events_corr)
                save_file, lst = self.save_anatuple_ttm(events_corr, Sel_Muon, Sel_Tau1, Sel_Tau2, self.tag, save_weightcorr=False)
                save_Event(save_file, lst, Treename)
        
        # JET corrections
        Jet_corr_list = ['JES', 'JER']
        for Jet_corr in Jet_corr_list:
            for val_corr in ['up','down']:
                Treename = 'Events_'+Jet_corr+'_'+self.period+'_'+val_corr
                print('JET corrections: saving '+ Treename)
                events_corr = self.Lepton_selection(events, Treename)
                events_corr, Sel_Muon, Sel_Tau1, Sel_Tau2 = self.analyse_ttm(events_corr)
                save_file, lst = self.save_anatuple_ttm(events_corr, Sel_Muon, Sel_Tau1, Sel_Tau2, self.tag, save_weightcorr=False)
                save_Event(save_file, lst, Treename)

        return out

    def analyse_ttm(self, events, out= None):
        # If out is None, do note save the cutflow

        # select ttm events: require at least 2 reco tau and 1 reco mu
        events_ttm = events[(ak.num(events.SelMuon) >= 1) & (ak.num(events.SelTau) >= 2)]

        if out != None:
            out[f'sumw_AtLeast2tau1mu'][self.ds] += ak.sum(events_ttm.genWeight)
            out[f'n_ev_AtLeast2tau1mu'][self.ds] += len(events_ttm)
            
        # veto events with more than one muon with pfRelIso03_all <= 0.15
        events_ttm = events_ttm[IsoMuon_mask(events_ttm, 1, iso_cut=0.4)]

        if out != None:
            out[f'sumw_NoAdditionalIsoMuon'][self.ds] += ak.sum(events_ttm.genWeight)
            out[f'n_ev_NoAdditionalIsoMuon'][self.ds] += len(events_ttm)

        # save info if an extra muon exist (with 0.15 <pfRelIso03_all < 0.4)
        events_ttm['nAdditionalMuon'] = ak.num(events_ttm.SelMuon[events_ttm.SelMuon.pfRelIso03_all > 0.15])

        # remove events with more than one isolated electron (assure orthogonality with tte tee tem)
        events_ttm = events_ttm[ak.num(events_ttm.SelElectron) == 0]

        if out != None:
            out[f'sumw_NoIsoElectron'][self.ds] += ak.sum(events_ttm.genWeight)
            out[f'n_ev_NoIsoElectron'][self.ds] += len(events_ttm)

        # events should pass most efficient HLT (for now only 1: IsoMu)
        if self.period == '2018':
            events_ttm = events_ttm[events_ttm.HLT.IsoMu24]
        if self.period == '2017':
            events_ttm = events_ttm[events_ttm.HLT.IsoMu27]
        if self.period == '2016':
            events_ttm = events_ttm[events_ttm.HLT.IsoMu24]
        if self.period == '2016_HIPM':
            events_ttm = events_ttm[events_ttm.HLT.IsoMu24]

        if out != None:
            out[f'sumw_HLT'][self.ds] += ak.sum(events_ttm.genWeight)
            out[f'n_ev_HLT'][self.ds] += len(events_ttm)

        events_ttm, Sel_Muon = Trigger_Muon_sel(events_ttm, self.period)

        if out != None:
            out[f'sumw_MuTriggerMatching'][self.ds] += ak.sum(events_ttm.genWeight)
            out[f'n_ev_MuTriggerMatching'][self.ds] += len(events_ttm)

        delta_r_cut = 0.5

        # select Tau candidate with dr(muon,Tau)>0.5 
        Tau1_candidate = events_ttm.SelTau[(delta_r(Sel_Muon,events_ttm.SelTau)> delta_r_cut)]

        #make sure at least 1 satisfy this condition
        events_ttm = events_ttm[ak.num(Tau1_candidate) >= 1]
        Sel_Muon = Sel_Muon[ak.num(Tau1_candidate) >= 1]
        Tau1_candidate = Tau1_candidate[ak.num(Tau1_candidate) >= 1]

        #Tau1 is the one with the highest isolation VSJet
        Sel_Tau1 = Tau1_candidate[ak.max(Tau1_candidate.rawDeepTau2018v2p5VSjet, axis=-1) == Tau1_candidate.rawDeepTau2018v2p5VSjet]
        Sel_Tau1 = Sel_Tau1[:,0]

        if out != None:
            out[f'sumw_drMuTau'][self.ds] += ak.sum(events_ttm.genWeight)
            out[f'n_ev_drMuTau'][self.ds] += len(events_ttm)

        # select Tau candidate with dr(muon,Tau)>0.5 and dr(Tau1,Tau)>0.5  and highest isolation VSJet
        events_ttm, Sel_Muon, Sel_Tau1, Sel_Tau2 = FinalTau_sel(events_ttm, Sel_Muon, Sel_Tau1, self.DeepTauVersion)

        if out != None:
            out[f'sumw_drTau2Leptons'][self.ds] += ak.sum(events_ttm.genWeight)
            out[f'n_ev_drTau2Leptons'][self.ds] += len(events_ttm)

        # Save bjets candidates
        bjet_candidates(events_ttm, Sel_Muon, Sel_Tau1, Sel_Tau2, self.period)

        #if len(events_ttm) == 0:
        #    print('0 events pass selection')
        #    return

        # Apply corrections for MC
        if self.mode != 'Data':
            events_ttm = self.compute_corrections_ttm(events_ttm, Sel_Muon, Sel_Tau1, Sel_Tau2)

        if out != None:
            out[f'sumw_corrections'][self.ds] += ak.sum(events_ttm.genWeight)
            out[f'n_ev_corrections'][self.ds] += len(events_ttm)

        return events_ttm, Sel_Muon, Sel_Tau1, Sel_Tau2

    def compute_corrections_ttm(self, events, Sel_Muon, Sel_Tau1, Sel_Tau2):
        #computing sf corrections        
        sf_tau1 = compute_sf_tau(Sel_Tau1, events, 'Tau1', self.period, self.DeepTauVersion)
        sf_tau2 = compute_sf_tau(Sel_Tau2, events, 'Tau2', self.period, self.DeepTauVersion)
        sf_mu = compute_sf_mu(Sel_Muon, events, 'Muon', self.period)
        Trigger_eff_corr_mu = get_trigger_correction_mu(Sel_Muon, events, 'Muon', self.period) # apply Trigger sf to the Muon that match HLT
        sf_L1PreFiring = compute_sf_L1PreFiring(events)
        PU_corr, PU_corr_up, PU_corr_down = get_pileup_correction(events, self.period)
        BTag_sf = get_BTag_sf(events, self.period)
        events['genWeight'] = events.genWeight * sf_tau1 * sf_tau2 * sf_mu * Trigger_eff_corr_mu * sf_L1PreFiring * PU_corr * BTag_sf
        return events
    
    def save_anatuple_ttm(self, events, Sel_Muon, Sel_Tau1, Sel_Tau2, tag, save_weightcorr=True):

        exclude_list = ['genPartIdx']
        
        save_file, lst = save_anatuple_common(self.ds, events, tag, self.period, 'ttm', save_weightcorr, self.sample_name)

        #info specific to the channel
        lst["nAdditionalMuon"] = np.array(events.nAdditionalMuon)
        lst['channelIndex'] = np.ones(len(events))*332

        if self.mode == 'signal':
            lst['HNLmass'] = np.ones(len(events))*int(self.ds[self.ds.rfind("-") + 1:])
        if self.mode != 'Data':
            Sel_Muon = add_gen_matching_info(events, Sel_Muon)
            Sel_Tau1  = add_gen_matching_info(events, Sel_Tau1)
            Sel_Tau2 = add_gen_matching_info(events, Sel_Tau2)

        #order tau by pt
        mask_ptmax = Sel_Tau1.pt >= Sel_Tau2.pt
        Sel_Tau_ptmax = ak.where(mask_ptmax, Sel_Tau1, Sel_Tau2)
        Sel_Tau_ptmin = ak.where(~mask_ptmax, Sel_Tau1, Sel_Tau2)

        lst = save_anatuple_lepton(events, Sel_Muon, lst, exclude_list, 'Muon')
        lst = save_anatuple_tau(events, Sel_Tau_ptmax, lst, exclude_list, self.mode, 'Tau1')
        lst = save_anatuple_tau(events, Sel_Tau_ptmin, lst, exclude_list, self.mode, 'Tau2')

        return save_file, lst

    def postprocess(self, accumulator):
        return accumulator