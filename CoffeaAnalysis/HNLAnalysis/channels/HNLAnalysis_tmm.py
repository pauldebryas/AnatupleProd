import numpy as np
import awkward as ak
from coffea import processor
from collections import defaultdict
import copy

from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_common, save_anatuple_lepton, save_anatuple_tau, save_bjets, save_Event, add_gen_matching_info
from CoffeaAnalysis.HNLAnalysis.correction_helpers import compute_sf_mu, compute_sf_tau, get_trigger_correction_mu, compute_sf_L1PreFiring, get_pileup_correction, get_BTag_sf
from CoffeaAnalysis.HNLAnalysis.helpers import IsoMuon_mask, Trigger_Muon_sel, FinalTau_sel, delta_r, bjet_candidates
from CoffeaAnalysis.HNLAnalysis.HNLProcessor import HNLProcessor

class HNLAnalysis_tmm(processor.ProcessorABC, HNLProcessor):
    def __init__(self, stitched_list, tag, xsecs, periods, dataHLT, debugMode):
        HNLProcessor.__init__(self, stitched_list, tag, xsecs, periods, dataHLT, debugMode)
        self.acc_dict = {}
        self.selections = self.get_selections()
        for selection in self.selections:
            self.acc_dict[f'n_ev_{selection}'] = defaultdict(int)
            self.acc_dict[f'sumw_{selection}'] = defaultdict(int)
        self._accumulator = self.acc_dict

        #the corresponding data sample for tmm channel (HLT=IsoMu24) 
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
            'AtLeast1tau2mu',
            'NoAdditionalIsoMuon',
            'NoIsoElectron',
            'HLT',
            'MuTriggerMatching',
            'drMu1Mu2',
            'drTauMuons',
            'corrections']

    # we will receive a NanoEvents
    def process(self, events):

        out = copy.deepcopy(self._accumulator)
        events, out = self.init_process(out, events)

        # cut specific for that channel
        #self.cut_e_iso = 0.15 # veto events with tight e isolation for ortogonality in signal region for channels with electrons 

        print('Running main analysis')
        # Do the general lepton selection
        events_tmm = self.Lepton_selection(events)

        # Apply the cuts and select leptons
        events_tmm, Sel_Muon, Sel_Muon2, Sel_Tau = self.analyse_tmm(events_tmm, out)

        # Save anatuple
        save_file, lst = self.save_anatuple_tmm(events_tmm, Sel_Muon, Sel_Muon2, Sel_Tau, self.tag, save_weightcorr=True)
        save_Event(save_file, lst, 'Events')
        if self.mode != 'Data':
            save_bjets(save_file, events_tmm) #for bTagSF

        if self.mode != 'Data':
            Tau_ES_corr_list = ['DM0', 'DM1', '3prong']
            # Compute Tau_ES for genuineTau
            for Tau_ES_corr in Tau_ES_corr_list:
                for val_corr in ['up','down']:
                    Treename = 'Events_GenuineTauES_'+Tau_ES_corr+'_'+self.period+'_'+val_corr
                    print('TAU ES corrections: saving '+ Treename)
                    events_corr = self.Lepton_selection(events, Treename)
                    events_corr, Sel_Muon, Sel_Muon2, Sel_Tau = self.analyse_tmm(events_corr)
                    save_file, lst = self.save_anatuple_tmm(events_corr, Sel_Muon, Sel_Muon2, Sel_Tau, self.tag, save_weightcorr=False)
                    save_Event(save_file, lst, Treename)

            # Compute Tau_ES for genuineElectron
            for Tau_ES_corr in Tau_ES_corr_list:
                for val_corr in ['up','down']:
                    Treename = 'Events_GenuineElectronES_'+Tau_ES_corr+'_'+self.period+'_'+val_corr
                    print('TAU ES corrections: saving '+ Treename)
                    events_corr = self.Lepton_selection(events, Treename)
                    events_corr, Sel_Muon, Sel_Muon2, Sel_Tau = self.analyse_tmm(events_corr)
                    save_file, lst = self.save_anatuple_tmm(events_corr, Sel_Muon, Sel_Muon2, Sel_Tau, self.tag, save_weightcorr=False)
                    save_Event(save_file, lst, Treename)

            # Compute Tau_ES for genuineMuon
            for val_corr in ['up','down']:
                Treename = 'Events_GenuineMuonES'+'_'+self.period+'_'+val_corr
                print('TAU ES corrections: saving '+ Treename)
                events_corr = self.Lepton_selection(events, Treename)
                events_corr, Sel_Muon, Sel_Muon2, Sel_Tau = self.analyse_tmm(events_corr)
                save_file, lst = self.save_anatuple_tmm(events_corr, Sel_Muon, Sel_Muon2, Sel_Tau, self.tag, save_weightcorr=False)
                save_Event(save_file, lst, Treename)

        # JET corrections
        Jet_corr_list = ['JES', 'JER']
        for Jet_corr in Jet_corr_list:
            for val_corr in ['up','down']:
                Treename = 'Events_'+Jet_corr+'_'+self.period+'_'+val_corr
                print('JET corrections: saving '+ Treename)
                events_corr = self.Lepton_selection(events, Treename)
                events_corr, Sel_Muon, Sel_Muon2, Sel_Tau = self.analyse_tmm(events_corr)
                save_file, lst = self.save_anatuple_tmm(events_corr, Sel_Muon, Sel_Muon2, Sel_Tau, self.tag, save_weightcorr=False)
                save_Event(save_file, lst, Treename)

        return out

    def analyse_tmm(self, events, out= None):
        # If out is None, do note save the cutflow

        # select tmm events: require at least 2 reco mu and 1 reco tau 
        events_tmm = events[(ak.num(events.SelMuon) >= 2) & (ak.num(events.SelTau) >= 1)]

        if out != None:
            out[f'sumw_AtLeast1tau2mu'][self.ds] += ak.sum(events_tmm.genWeight)
            out[f'n_ev_AtLeast1tau2mu'][self.ds] += len(events_tmm)

        # veto events with more than two muon with pfRelIso03_all <= 0.15
        events_tmm = events_tmm[IsoMuon_mask(events_tmm, 2, iso_cut=0.15)]

        if out != None:
            out[f'sumw_NoAdditionalIsoMuon'][self.ds] += ak.sum(events_tmm.genWeight)
            out[f'n_ev_NoAdditionalIsoMuon'][self.ds] += len(events_tmm)

        # save info if an extra muon exist with 0.15 <pfRelIso03_all < 0.4
        events_tmm['nAdditionalMuon'] = ak.num(events_tmm.SelMuon[events_tmm.SelMuon.pfRelIso03_all > 0.15])

        # remove events with more than one isolated electron (assure orthogonality with tte tee tem)
        events_tmm = events_tmm[ak.num(events_tmm.SelElectron) == 0]

        if out != None:
            out[f'sumw_NoIsoElectron'][self.ds] += ak.sum(events_tmm.genWeight)
            out[f'n_ev_NoIsoElectron'][self.ds] += len(events_tmm)

        # events should pass most efficient HLT (for now only 1: IsoMu)
        if self.period == '2018':
            events_tmm = events_tmm[events_tmm.HLT.IsoMu24]
        if self.period == '2017':
            events_tmm = events_tmm[events_tmm.HLT.IsoMu27]
        if self.period == '2016':
            events_tmm = events_tmm[events_tmm.HLT.IsoMu24]
        if self.period == '2016_HIPM':
            events_tmm = events_tmm[events_tmm.HLT.IsoMu24]

        if out != None:
            out[f'sumw_HLT'][self.ds] += ak.sum(events_tmm.genWeight)
            out[f'n_ev_HLT'][self.ds] += len(events_tmm)

         # Select trigger Muon
        events_tmm, Sel_Muon = Trigger_Muon_sel(events_tmm, self.period)

        if out != None:
            out[f'sumw_MuTriggerMatching'][self.ds] += ak.sum(events_tmm.genWeight)
            out[f'n_ev_MuTriggerMatching'][self.ds] += len(events_tmm)

        delta_r_cut = 0.5

        # select Muon2, that should be away from Muon1:  dr(Mu2,Mu1)>0.5 & pt>15GeV
        Mu2_candidate = events_tmm.SelMuon[(delta_r(Sel_Muon,events_tmm.SelMuon) > delta_r_cut)]

        #make sure at least 1 satisfy this condition
        events_tmm = events_tmm[ak.num(Mu2_candidate) >= 1]
        Sel_Muon = Sel_Muon[ak.num(Mu2_candidate) >= 1]
        Mu2_candidate = Mu2_candidate[ak.num(Mu2_candidate) >= 1]
        
        #Mu2 is the one with the minimum pfRelIso03_all
        Sel_Muon2 = Mu2_candidate[ak.min(Mu2_candidate.pfRelIso03_all, axis=-1) == Mu2_candidate.pfRelIso03_all]
        Sel_Muon2 = Sel_Muon2[:,0]

        if out != None:
            out[f'sumw_drMu1Mu2'][self.ds] += ak.sum(events_tmm.genWeight)
            out[f'n_ev_drMu1Mu2'][self.ds] += len(events_tmm)

        # select Tau candidate with dr(muon,Tau)>0.5 and dr(muon2,Tau)>0.5  and highest isolation VSJet
        events_tmm, Sel_Muon, Sel_Muon2, Sel_Tau = FinalTau_sel(events_tmm, Sel_Muon, Sel_Muon2, self.DeepTauVersion)

        if out != None:
            out[f'sumw_drTauMuons'][self.ds] += ak.sum(events_tmm.genWeight)
            out[f'n_ev_drTauMuons'][self.ds] += len(events_tmm)

        # Save bjets candidates
        bjet_candidates(events_tmm, Sel_Muon, Sel_Muon2, Sel_Tau, self.period)

        # if len(events_tmm) == 0:
        #     print('0 events pass selection')
        #     return

        # Apply corrections for MC
        if self.mode != 'Data':
            events_tmm = self.compute_corrections_tmm(events_tmm, Sel_Muon, Sel_Muon2, Sel_Tau)

        if out != None:
            out[f'sumw_corrections'][self.ds] += ak.sum(events_tmm.genWeight)
            out[f'n_ev_corrections'][self.ds] += len(events_tmm)
        
        return events_tmm, Sel_Muon, Sel_Muon2, Sel_Tau

    def compute_corrections_tmm(self, events, Sel_Muon, Sel_Muon2, Sel_Tau):
        #computing sf corrections        
        sf_tau = compute_sf_tau(Sel_Tau, events, 'Tau', self.period, self.DeepTauVersion)
        sf_mu1 = compute_sf_mu(Sel_Muon, events, 'Muon1', self.period)
        sf_mu2 = compute_sf_mu(Sel_Muon2, events, 'Muon2', self.period)
        Trigger_eff_corr_mu = get_trigger_correction_mu(Sel_Muon, events, 'Muon1', self.period) # apply Trigger sf to the Muon that match HLT
        Trigger_eff_corr_mu2 = get_trigger_correction_mu(Sel_Muon2, events, 'Muon2', self.period) # compute Trigger sf for Muon2
        sf_L1PreFiring = compute_sf_L1PreFiring(events)
        PU_corr, PU_corr_up, PU_corr_down = get_pileup_correction(events, self.period)
        BTag_sf = get_BTag_sf(events, self.period)
        events['genWeight'] = events.genWeight * sf_tau * sf_mu1 * sf_mu2 * Trigger_eff_corr_mu * sf_L1PreFiring * PU_corr * BTag_sf
        return events
    
    def save_anatuple_tmm(self, events, Sel_Muon, Sel_Muon2, Sel_Tau, tag, save_weightcorr=True):

        exclude_list = ['genPartIdx']
        
        save_file, lst = save_anatuple_common(self.ds, events, tag, self.period, 'tmm', save_weightcorr)

        #info specific to the channel
        lst["nAdditionalMuon"] = np.array(events.nAdditionalMuon)
        lst['channelIndex'] = np.ones(len(events))*322

        if self.mode == 'signal':
            lst['HNLmass'] = np.ones(len(events))*int(self.ds[self.ds.rfind("-") + 1:])
        # save GenPart info in case MC sample
        if self.mode != 'Data':
            Sel_Muon = add_gen_matching_info(events, Sel_Muon)
            Sel_Muon2  = add_gen_matching_info(events, Sel_Muon2)
            Sel_Tau = add_gen_matching_info(events, Sel_Tau)

        #order muon by pt
        mask_ptmax = Sel_Muon.pt >= Sel_Muon2.pt
        Sel_Muon_ptmax = ak.where(mask_ptmax, Sel_Muon, Sel_Muon2)
        Sel_Muon_ptmin = ak.where(~mask_ptmax, Sel_Muon, Sel_Muon2)

        lst = save_anatuple_lepton(Sel_Muon_ptmax, lst, exclude_list, 'Muon1')
        lst = save_anatuple_lepton(Sel_Muon_ptmin, lst, exclude_list, 'Muon2')
        lst = save_anatuple_tau(events, Sel_Tau, lst, exclude_list, self.mode, 'Tau')

        return save_file, lst

    def postprocess(self, accumulator):
        return accumulator