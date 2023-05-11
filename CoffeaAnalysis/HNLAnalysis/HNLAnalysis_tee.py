import numpy as np
import awkward as ak
from coffea import processor


from CoffeaAnalysis.HNLAnalysis.helpers import save_anatuple_common, save_anatuple_lepton, save_anatuple_tau, save_Event_bjets
from CoffeaAnalysis.HNLAnalysis.correction_helpers import compute_sf_e, compute_sf_tau, get_trigger_correction_e
from CoffeaAnalysis.HNLAnalysis.helpers import IsoElectron_mask, Ele32_Electron_sel, FinalTau_sel, delta_r, bjet_candidates
from CoffeaAnalysis.HNLAnalysis.HNLProcessor import HNLProcessor

class HNLAnalysis_tee(processor.ProcessorABC, HNLProcessor):
    def __init__(self, stitched_list, tag, xsecs):
        HNLProcessor.__init__(self, stitched_list, tag, xsecs)
        acc_dict = {}
        self.selections = self.get_selections()
        for selection in self.selections:
            acc_dict[f'n_ev_{selection}'] = processor.defaultdict_accumulator(int)
            acc_dict[f'sumw_{selection}'] = processor.defaultdict_accumulator(float)
        self._accumulator = processor.dict_accumulator(acc_dict)

        #the corresponding data sample for tee channel (HLT=Ele32)
        self.dataHLT = 'EGamma_2018'

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
        events, out, ds, mode = self.init_process(out, events)

        # cut specific for that channel 
        self.cut_mu_iso = 0.15 # veto events with tight mu isolation for ortogonality in signal region for channels with muon
        self.cut_tau_idVSe = 6 # require tight isolation against electron for channel with reco electron

        events = self.Lepton_selection(events, mode)

        self.analyse_tee(events, out, ds, mode)

        return out

    def analyse_tee(self, events, out, ds, mode):

        # select tee events: require 2 reco e and 1 reco tau 
        events_tee = events[(ak.num(events.SelElectron) >= 2) & (ak.num(events.SelTau) >= 1)]

        out[f'sumw_AtLeast1tau2e'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_AtLeast1tau2e'][ds] += len(events_tee)

        # veto events with more than two electrons with pfRelIso03_all <= 0.15
        events_tee = events_tee[ IsoElectron_mask(events_tee, 2, iso_cut=0.15)]

        out[f'sumw_NoAdditionalIsoElectron'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_NoAdditionalIsoElectron'][ds] += len(events_tee)

        # save info if an extra electron exist (with 0.15 <pfRelIso03_all < 0.4)
        events_tee['nAdditionalElectron'] = ak.num(events_tee.SelElectron[events_tee.SelMuon.pfRelIso03_all > 0.15])

        # remove events with more than one isolated muon (assure orthogonality with ttm tmm tem)
        events_tee = events_tee[ak.num(events_tee.SelMuon) == 0]

        out[f'sumw_noIsoMuon'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_noIsoMuon'][ds] += len(events_tee)

        # events should pass most efficient HLT (Ele32_WPTight_Gsf_L1DoubleEG)
        events_tee = events_tee[events_tee.HLT.Ele32_WPTight_Gsf_L1DoubleEG]

        out[f'sumw_HLT'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_HLT'][ds] += len(events_tee)

        #find electron with minimum pfRelIso03_all, and matching Ele32 trigger. In case same isolation, we take the first one by default
        events_tee, Sel_Electron = Ele32_Electron_sel(events_tee)

        out[f'sumw_eSelection'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_eSelection'][ds] += len(events_tee) 

        # select Electron2, that should be away from first Electron: dr(e2,e)>0.5
        delta_r_cut = 0.5
        e2_candidate = events_tee.SelElectron[(delta_r(Sel_Electron,events_tee.SelElectron)> delta_r_cut)]

        #make sure at least 1 satisfy this condition
        events_tee = events_tee[ak.num(e2_candidate) >= 1]
        Sel_Electron = Sel_Electron[ak.num(e2_candidate) >= 1]
        e2_candidate = e2_candidate[ak.num(e2_candidate) >= 1]

        #e2 is the one with the minimum pfRelIso03_all
        Sel_Electron2 = e2_candidate[ak.min(e2_candidate.pfRelIso03_all, axis=-1) == e2_candidate.pfRelIso03_all]
        Sel_Electron2 = Sel_Electron2[:,0]

        out[f'sumw_e2sel'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_e2sel'][ds] += len(events_tee)

        # select Tau candidate with dr(e1,Tau)>0.5 and dr(e2,Tau)>0.5 and highest isolation VSJet
        events_tee, Sel_Electron, Sel_Electron2, Sel_Tau = FinalTau_sel(events_tee, Sel_Electron, Sel_Electron2)

        out[f'sumw_tausel'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_tausel'][ds] += len(events_tee)
        
        bjet_candidate = bjet_candidates(events_tee, Sel_Electron, Sel_Electron2, Sel_Tau)

        events_tee['nbjets'] = ak.num(bjet_candidate)
        events_tee['bjets'] = bjet_candidate

        if len(events_tee) == 0:
            print('0 events pass selection')
            return

       #computing corrections
        if mode != 'Data':
            sf_tau = compute_sf_tau(Sel_Tau)
            sf_e1 = compute_sf_e(Sel_Electron)
            sf_e2 = compute_sf_e(Sel_Electron2)
            Trigger_eff_corr_e1 = get_trigger_correction_e(Sel_Electron)
            events_tee.genWeight = events_tee.genWeight * sf_tau * sf_e1 * sf_e2 * Trigger_eff_corr_e1

        out[f'sumw_corrections'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_corrections'][ds] += len(events_tee)

        # Save anatuple
        self.save_anatuple_tee(ds, events_tee, Sel_Electron, Sel_Electron2, Sel_Tau, self.tag, mode)

        return events_tee

    def save_anatuple_tee(self, ds, events, Sel_Electron, Sel_Electron2, Sel_Tau, tag, mode):

        exclude_list = ['genPartIdx']
        
        save_file, lst = save_anatuple_common(ds, events, tag)

        #info specific to the channel
        lst["nAdditionalElectron"] = np.array(events.nAdditionalElectron)
        lst['channelIndex'] = np.ones(len(events))*311

        if mode == 'signal':
            if 'HNL_tau_M-' not in ds:
                raise 'signal samples not starting with HNL_tau_M-'
            lst['HNLmass'] = np.ones(len(events))*int(ds[len('HNL_tau_M-'):])
        
        lst = save_anatuple_lepton(Sel_Electron, lst, exclude_list, 'Electron1')
        lst = save_anatuple_lepton(Sel_Electron2, lst, exclude_list, 'Electron2')
        lst = save_anatuple_tau(events, Sel_Tau, lst, exclude_list, mode, 'Tau')

        save_Event_bjets(save_file, lst, events)

        return

    def postprocess(self, accumulator):
        return accumulator
