import numpy as np
import awkward as ak
from coffea import processor, hist
from CoffeaAnalysis.helpers import delta_r, data_goodrun_lumi

class HNLAnalysis(processor.ProcessorABC):
    def __init__(self):
        ds_axis = hist.Cat("ds", "Primary dataset")
        acc_dict = {var: hist.Hist("Counts", ds_axis, axis) for var, axis in self.get_var_axis_pairs()}
        acc_dict[f'n_ev_all'] = processor.defaultdict_accumulator(int)
        acc_dict[f'sumw_all'] = processor.defaultdict_accumulator(float)
        self.selections = self.get_selections()
        self.channels = self.get_channels()
        for selection in self.selections:
            for channel in self.channels:
                acc_dict[f'n_ev_{selection}_{channel}'] = processor.defaultdict_accumulator(int)
                acc_dict[f'sumw_{selection}_{channel}'] = processor.defaultdict_accumulator(float)

        acc_dict['sel_array'] = processor.column_accumulator(np.ndarray((0, 12)))
        self._accumulator = processor.dict_accumulator(acc_dict)
    
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
            'bjetveto',
            'chargeveto',
            'metselection',
            'zveto'
        ]
    
    @staticmethod
    def get_channels():
        return [
            'tte',
            'ttm',
            'tee',
            'tmm',
            'tem_SS',
            'tem_OS',
            'ttt'
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
        ]

        v_a_pairs_tags = []
        tags = ['_tte','_ttm','_tee','_tmm','_tem_SS','_tem_OS','_ttt']
        for tag in tags:
            v_a_pairs_tags.extend([(name+tag, axis) for name, axis in v_a_pairs])

        return v_a_pairs_tags
    
    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        ds = events.metadata["dataset"] # dataset name

        print("For dataset : "+ ds)

        mode ='MCbackground'
        if 'Data' in ds:
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

        # Reco event selection: minimal requirement for leptons
        #cut to apply
        #tau
        cut_tau_pt = 20. # Tau_pt > cut_tau_pt
        cut_tau_eta = 2.3 #abs(Tau_eta) < cut_tau_eta
        cut_tau_idVSmu = 2 # Tau_idDeepTau2017v2p1VSmu >= Loose
        cut_tau_idVSe = 4 # Tau_idDeepTau2017v2p1VSe >= VLoose
        cut_tau_idVSjet = 16 # Tau_idDeepTau2017v2p1VSjet >= Medium
        #electrons
        cut_e_pt = 25. # Electron_pt > cut_e_pt
        cut_e_eta = 2.5 # abs(Electron_eta) < cut_e_eta
        cut_e_id = 0 # Electron_mvaFall17V2Iso_WP90 > cut_e_id
        #muons
        cut_mu_pt = 25. # Muon_pt > cut_mu_pt
        cut_mu_eta = 2.4 # abs(Muon_eta) < cut_mu_eta
        cut_mu_id = 0 #(Muon_mediumId > cut_mu_id || Muon_tightId > cut_mu_id)
        cut_mu_iso = 0.2 # Muon_pfRelIso03_all < cut_mu_iso

        #lepton selection (Reco level)
        events['SelTau'] = events.Tau[(events.Tau.pt > cut_tau_pt) & (np.abs(events.Tau.eta) < cut_tau_eta) & (events.Tau.idDeepTau2017v2p1VSmu >= cut_tau_idVSmu) & (events.Tau.idDeepTau2017v2p1VSjet >= cut_tau_idVSjet) & (events.Tau.idDeepTau2017v2p1VSe >= cut_tau_idVSe)]
        events['SelElectron'] = events.Electron[(events.Electron.pt > cut_e_pt) & (np.abs(events.Electron.eta) < cut_e_eta) & (events.Electron.mvaFall17V2Iso_WP90 > cut_e_id)]
        events['SelMuon'] = events.Muon[(events.Muon.pt > cut_mu_pt) & (np.abs(events.Muon.eta) < cut_mu_eta) & ((events.Muon.mediumId > cut_mu_id) | (events.Muon.tightId > cut_mu_id)) & (events.Muon.pfRelIso03_all < cut_mu_iso)]

        if mode == 'Data':
            if 'Data_SingleMuon' in ds:
                print("Analysis of ttm tmm and tem(SS/OS) channel (IsoMu24 HLT)")
                self.analyse_ttm(events, out, ds)
                self.analyse_tmm(events, out, ds)
                self.analyse_tem_SS(events, out, ds)
                self.analyse_tem_OS(events, out, ds)
            if 'Data_EGamma' in ds:
                print("Analysis of tee channel (Ele32 HLT)")
                self.analyse_tee(events, out, ds)
            if 'Data_Tau' in ds:
                print("Analysis of tte channel (DoubleTau HLT)")
                #self.analyse_ttt(events, out, ds)
                self.analyse_tte(events, out, ds)
        else:
            print("Analysis of all channels")
            self.analyse_ttm(events, out, ds)
            self.analyse_tmm(events, out, ds)
            self.analyse_tem_SS(events, out, ds)
            self.analyse_tem_OS(events, out, ds)
            #self.analyse_ttt(events, out, ds)
            self.analyse_tte(events, out, ds)
            self.analyse_tee(events, out, ds)

        return out

    def saved_leading_muon(self, events, Sel_Muon, out, ds, tag=''):
        #print("save_muon " + ds)
        out[f'pt_mu1_{tag}'].fill(ds=ds, pt=ak.flatten(Sel_Muon.pt, axis=None), weight=events.genWeight)
        out[f'eta_mu1_{tag}'].fill(ds=ds, eta=ak.flatten(Sel_Muon.eta, axis=None), weight=events.genWeight)
        out[f'phi_mu1_{tag}'].fill(ds=ds, phi=ak.flatten(Sel_Muon.phi, axis=None), weight=events.genWeight)
        out[f'mass_mu1_{tag}'].fill(ds=ds, mass=ak.flatten(Sel_Muon.mass, axis=None), weight=events.genWeight)

    def saved_subleading_muon(self, events, Sel_Muon, out, ds, tag=''):
        #print("save_muon " + ds)
        out[f'pt_mu2_{tag}'].fill(ds=ds, pt=ak.flatten(Sel_Muon.pt, axis=None), weight=events.genWeight)
        out[f'eta_mu2_{tag}'].fill(ds=ds, eta=ak.flatten(Sel_Muon.eta, axis=None), weight=events.genWeight)
        out[f'phi_mu2_{tag}'].fill(ds=ds, phi=ak.flatten(Sel_Muon.phi, axis=None), weight=events.genWeight)
        out[f'mass_mu2_{tag}'].fill(ds=ds, mass=ak.flatten(Sel_Muon.mass, axis=None), weight=events.genWeight)

    def saved_leading_electron(self, events, Sel_Electron, out, ds, tag=''):
        #print("save_electron " + ds)
        out[f'pt_e1_{tag}'].fill(ds=ds, pt=ak.flatten(Sel_Electron.pt, axis=None), weight=events.genWeight)
        out[f'eta_e1_{tag}'].fill(ds=ds, eta=ak.flatten(Sel_Electron.eta, axis=None), weight=events.genWeight)
        out[f'phi_e1_{tag}'].fill(ds=ds, phi=ak.flatten(Sel_Electron.phi, axis=None), weight=events.genWeight)
        out[f'mass_e1_{tag}'].fill(ds=ds, mass=ak.flatten(Sel_Electron.mass, axis=None), weight=events.genWeight)

    def saved_subleading_electron(self, events, Sel_Electron, out, ds, tag=''):
        #print("save_electron " + ds)
        out[f'pt_e2_{tag}'].fill(ds=ds, pt=ak.flatten(Sel_Electron.pt, axis=None), weight=events.genWeight)
        out[f'eta_e2_{tag}'].fill(ds=ds, eta=ak.flatten(Sel_Electron.eta, axis=None), weight=events.genWeight)
        out[f'phi_e2_{tag}'].fill(ds=ds, phi=ak.flatten(Sel_Electron.phi, axis=None), weight=events.genWeight)
        out[f'mass_e2_{tag}'].fill(ds=ds, mass=ak.flatten(Sel_Electron.mass, axis=None), weight=events.genWeight)

    def saved_leading_tau(self, events, Sel_Tau, out, ds, tag=''):
        #print("save_tau " + ds)
        out[f'pt_tau1_{tag}'].fill(ds=ds, pt=ak.flatten(Sel_Tau.pt, axis=None), weight=events.genWeight)
        out[f'eta_tau1_{tag}'].fill(ds=ds, eta=ak.flatten(Sel_Tau.eta, axis=None), weight=events.genWeight)
        out[f'phi_tau1_{tag}'].fill(ds=ds, phi=ak.flatten(Sel_Tau.phi, axis=None), weight=events.genWeight)
        out[f'mass_tau1_{tag}'].fill(ds=ds, mass=ak.flatten(Sel_Tau.mass, axis=None), weight=events.genWeight)

    def saved_subleading_tau(self, events, Sel_Tau, out, ds, tag=''):
        #print("save_tau " + ds)
        out[f'pt_tau2_{tag}'].fill(ds=ds, pt=ak.flatten(Sel_Tau.pt, axis=None), weight=events.genWeight)
        out[f'eta_tau2_{tag}'].fill(ds=ds, eta=ak.flatten(Sel_Tau.eta, axis=None), weight=events.genWeight)
        out[f'phi_tau2_{tag}'].fill(ds=ds, phi=ak.flatten(Sel_Tau.phi, axis=None), weight=events.genWeight)
        out[f'mass_tau2_{tag}'].fill(ds=ds, mass=ak.flatten(Sel_Tau.mass, axis=None), weight=events.genWeight)

    def saved_subsubleading_tau(self, events, Sel_Tau, out, ds, tag=''):
        #print("save_tau " + ds)
        out[f'pt_tau3_{tag}'].fill(ds=ds, pt=ak.flatten(Sel_Tau.pt, axis=None), weight=events.genWeight)
        out[f'eta_tau3_{tag}'].fill(ds=ds, eta=ak.flatten(Sel_Tau.eta, axis=None), weight=events.genWeight)
        out[f'phi_tau3_{tag}'].fill(ds=ds, phi=ak.flatten(Sel_Tau.phi, axis=None), weight=events.genWeight)
        out[f'mass_tau3_{tag}'].fill(ds=ds, mass=ak.flatten(Sel_Tau.mass, axis=None), weight=events.genWeight)

    def saved_dilepton_mass(self, events, Lepton1, Lepton2, out, ds, tag=''):
        #print("dilepton_mass " + ds)
        out[f'comb_mass_l1l2_{tag}'].fill(ds=ds, mc=ak.flatten((Lepton1 + Lepton2).mass, axis=None), weight=events.genWeight)

    def saved_dilepton_mass_taul1_OS(self, events, Lepton, Tau1, Tau2, out, ds, tag=''):
        # +++ and --- events not recorded!

        #Lepton and Tau1 OS (and tau2 SS as Lepton)
        sel1 = ak.flatten((Tau1.charge != Tau2.charge) & (Lepton.charge != Tau1.charge))
        out[f'comb_mass_taul1_{tag}'].fill(ds=ds, mc=ak.flatten((Lepton[sel1] + Tau1[sel1]).mass, axis=None), weight=events[sel1].genWeight)
        #Lepton and Tau2 OS (and tau1 SS as Lepton)
        sel2 = ak.flatten((Tau1.charge != Tau2.charge) & (Lepton.charge != Tau2.charge))
        out[f'comb_mass_taul1_{tag}'].fill(ds=ds, mc=ak.flatten((Lepton[sel2] + Tau2[sel2]).mass, axis=None), weight=events[sel2].genWeight)
        #Lepton and Tau2/Tau1 OS --> take the leading tau
        sel3 = ak.flatten((Tau1.charge == Tau2.charge) & (Lepton.charge != Tau2.charge) & (Tau1.pt >= Tau2.pt))
        out[f'comb_mass_taul1_{tag}'].fill(ds=ds, mc=ak.flatten((Lepton[sel3] + Tau1[sel3]).mass, axis=None), weight=events[sel3].genWeight)
        sel4 = ak.flatten((Tau1.charge == Tau2.charge) & (Lepton.charge != Tau2.charge) & (Tau1.pt < Tau2.pt))
        out[f'comb_mass_taul1_{tag}'].fill(ds=ds, mc=ak.flatten((Lepton[sel4] + Tau2[sel4]).mass, axis=None), weight=events[sel4].genWeight)

        #check orthogonality/sanity check
        sel5 = ak.flatten((Tau1.charge == Tau2.charge) & (Lepton.charge == Tau2.charge))
        if (len(events) - (ak.sum(sel1)+ak.sum(sel2)+ak.sum(sel3)+ak.sum(sel4)+ak.sum(sel5))) != 0 :
            print('problem with saved_dilepton_mass_taul1_OS for ds:' + ds)

    def saved_drl1l2(self, events, Lepton1, Lepton2, out, ds, tag=''):
        #print("save_drl1l2 " + ds)
        out[f'dr_l1l2_{tag}'].fill(ds=ds, dr=ak.flatten(delta_r(Lepton1,Lepton2), axis=None), weight=events.genWeight)

    def saved_MET(self, events, out, ds, tag=''):
        #print("save_drl1l2 " + ds)
        out[f'met_{tag}'].fill(ds=ds, met=events.MET.pt, weight=events.genWeight)




    def analyse_ttm(self, events, out, ds):

        tag = 'ttm'
        # select ttm events: require 2 reco tau and 1 reco mu
        events_ttm = events[(ak.num(events.SelMuon) == 1) & (ak.num(events.SelTau) == 2)]

        out[f'sumw_3leptons_{tag}'][ds] += ak.sum(events_ttm.genWeight)
        out[f'n_ev_3leptons_{tag}'][ds] += len(events_ttm)

        # events should pass most efficient HLT (for now only 1: IsoMu24)
        events_ttm = events_ttm[events_ttm.HLT.IsoMu24]

        out[f'sumw_hlt_{tag}'][ds] += ak.sum(events_ttm.genWeight)
        out[f'n_ev_hlt_{tag}'][ds] += len(events_ttm)

        if len(events_ttm) == 0:
            return

        # select reco muon that match HLT IsoMu24
        Sel_Muon = self.select_lep1_IsoMu24(events_ttm, min_dr_cut=0.2)

        #removing non matching events
        events_ttm = events_ttm[(ak.num(Sel_Muon) == 1)]
        Sel_Muon = Sel_Muon[(ak.num(Sel_Muon) == 1)]

        out[f'sumw_l1sel_{tag}'][ds] += ak.sum(events_ttm.genWeight)
        out[f'n_ev_l1sel_{tag}'][ds] += len(events_ttm)

        # select reco Tau1 with dr(tau1,mu)>0.5 (in case there is more than 1 selected we choose the one with higher pt )
        Sel_Tau1 = self.select_lep2(events_ttm, Sel_Muon, type='tau', delta_r_cut = 0.5)

        #removing non matching events
        Sel_Muon = Sel_Muon[(ak.num(Sel_Tau1) == 1)]
        events_ttm = events_ttm[(ak.num(Sel_Tau1) == 1)]
        Sel_Tau1 = Sel_Tau1[(ak.num(Sel_Tau1) == 1)]

        out[f'sumw_l2sel_{tag}'][ds] += ak.sum(events_ttm.genWeight)
        out[f'n_ev_l2sel_{tag}'][ds] += len(events_ttm)

        # select reco Tau2 with dr(tau2,mu)>0.5 and dr(tau2,tau1)>0.5 (in case there is more than 1 selected we choose the one with higher pt )
        Sel_Tau2 = self.select_lep3(events_ttm, Sel_Muon, Sel_Tau1, type='tau', delta_r_cut = 0.5)

        #removing non matching events
        Sel_Muon = Sel_Muon[(ak.num(Sel_Tau2) == 1)]
        Sel_Tau1 = Sel_Tau1[(ak.num(Sel_Tau2) == 1)]
        events_ttm = events_ttm[(ak.num(Sel_Tau2) == 1)]
        Sel_Tau2 = Sel_Tau2[(ak.num(Sel_Tau2) == 1)]
        
        out[f'sumw_l3sel_{tag}'][ds] += ak.sum(events_ttm.genWeight)
        out[f'n_ev_l3sel_{tag}'][ds] += len(events_ttm)

        events_ttm, Sel_Tau1, Sel_Tau2, Sel_Muon = self.bjet_veto(events_ttm, Sel_Tau1, Sel_Tau2, Sel_Muon)
        out[f'sumw_bjetveto_{tag}'][ds] += ak.sum(events_ttm.genWeight)
        out[f'n_ev_bjetveto_{tag}'][ds] += len(events_ttm)

        events_ttm, Sel_Tau1, Sel_Tau2, Sel_Muon = self.charge_veto(events_ttm, Sel_Tau1, Sel_Tau2, Sel_Muon)
        out[f'sumw_chargeveto_{tag}'][ds] += ak.sum(events_ttm.genWeight)
        out[f'n_ev_chargeveto_{tag}'][ds] += len(events_ttm)

        events_ttm, Sel_Tau1, Sel_Tau2, Sel_Muon = self.met_veto(events_ttm, Sel_Tau1, Sel_Tau2, Sel_Muon)
        out[f'sumw_metselection_{tag}'][ds] += ak.sum(events_ttm.genWeight)
        out[f'n_ev_metselection_{tag}'][ds] += len(events_ttm)

        events_ttm, Sel_Tau1, Sel_Tau2, Sel_Muon = self.z_veto_ttl(events_ttm, Sel_Tau1, Sel_Tau2, Sel_Muon)
        out[f'sumw_zveto_{tag}'][ds] += ak.sum(events_ttm.genWeight)
        out[f'n_ev_zveto_{tag}'][ds] += len(events_ttm)

        # Save histograms
        self.saved_leading_muon(events_ttm, Sel_Muon, out, ds, tag)
        self.saved_leading_tau(events_ttm, Sel_Tau1, out, ds, tag)
        self.saved_subleading_tau(events_ttm, Sel_Tau2, out, ds, tag)
        self.saved_dilepton_mass_taul1_OS(events_ttm, Sel_Muon, Sel_Tau1, Sel_Tau2, out, ds, tag)
        self.saved_MET(events_ttm, out, ds, tag)
        self.saved_drl1l2(events_ttm, Sel_Muon, Sel_Tau1, out, ds, tag)

        return events_ttm

    def analyse_tmm(self, events, out, ds):

        tag = 'tmm'
        # select tmm events: require 2 reco mu and 1 reco tau 
        events_tmm = events[(ak.num(events.SelMuon) == 2) & (ak.num(events.SelTau) == 1)]

        out[f'sumw_3leptons_{tag}'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_3leptons_{tag}'][ds] += len(events_tmm)

        # events should pass most efficient HLT (for now only 1: IsoMu24)
        events_tmm = events_tmm[events_tmm.HLT.IsoMu24]

        out[f'sumw_hlt_{tag}'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_hlt_{tag}'][ds] += len(events_tmm)

        if len(events_tmm) == 0:
            return

        # select reco muon that match HLT IsoMu24
        Sel_Muon = self.select_lep1_IsoMu24(events_tmm, min_dr_cut=0.2)

        #removing non matching events
        events_tmm = events_tmm[(ak.num(Sel_Muon) == 1)]
        Sel_Muon = Sel_Muon[(ak.num(Sel_Muon) == 1)]

        out[f'sumw_l1sel_{tag}'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_l1sel_{tag}'][ds] += len(events_tmm)

        # select reco Muon2 with dr(mu2,mu)>0.5 (in case there is more than 1 selected we choose the one with higher pt )
        Sel_Muon2 = self.select_lep2(events_tmm, Sel_Muon, type='muon', delta_r_cut = 0.5)

        #removing non matching events
        Sel_Muon = Sel_Muon[(ak.num(Sel_Muon2) == 1)]
        events_tmm = events_tmm[(ak.num(Sel_Muon2) == 1)]
        Sel_Muon2 = Sel_Muon2[(ak.num(Sel_Muon2) == 1)]

        out[f'sumw_l2sel_{tag}'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_l2sel_{tag}'][ds] += len(events_tmm)

        # select reco Tau with dr(tau,mu)>0.5 and dr(tau,muon2)>0.5 (in case there is more than 1 selected we choose the one with higher pt )
        Sel_Tau = self.select_lep3(events_tmm, Sel_Muon, Sel_Muon2, type='tau', delta_r_cut = 0.5)

        #removing non matching events
        Sel_Muon = Sel_Muon[(ak.num(Sel_Tau) == 1)]
        Sel_Muon2 = Sel_Muon2[(ak.num(Sel_Tau) == 1)]
        events_tmm = events_tmm[(ak.num(Sel_Tau) == 1)]
        Sel_Tau = Sel_Tau[(ak.num(Sel_Tau) == 1)]
        
        out[f'sumw_l3sel_{tag}'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_l3sel_{tag}'][ds] += len(events_tmm)

        events_tmm, Sel_Tau, Sel_Muon, Sel_Muon2 = self.bjet_veto(events_tmm, Sel_Tau, Sel_Muon, Sel_Muon2)
        out[f'sumw_bjetveto_{tag}'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_bjetveto_{tag}'][ds] += len(events_tmm)

        events_tmm, Sel_Tau, Sel_Muon, Sel_Muon2 = self.charge_veto(events_tmm, Sel_Tau, Sel_Muon, Sel_Muon2)
        out[f'sumw_chargeveto_{tag}'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_chargeveto_{tag}'][ds] += len(events_tmm)

        events_tmm, Sel_Tau, Sel_Muon, Sel_Muon2 = self.met_veto(events_tmm, Sel_Tau, Sel_Muon, Sel_Muon2)
        out[f'sumw_metselection_{tag}'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_metselection_{tag}'][ds] += len(events_tmm)

        events_tmm, Sel_Tau, Sel_Muon, Sel_Muon2 = self.z_veto_tll(events_tmm, Sel_Tau, Sel_Muon, Sel_Muon2)
        out[f'sumw_zveto_{tag}'][ds] += ak.sum(events_tmm.genWeight)
        out[f'n_ev_zveto_{tag}'][ds] += len(events_tmm)

        # Save histograms
        self.saved_leading_muon(events_tmm, Sel_Muon, out, ds, tag)
        self.saved_subleading_muon(events_tmm, Sel_Muon2, out, ds, tag)
        self.saved_leading_tau(events_tmm, Sel_Tau, out, ds, tag)
        self.saved_dilepton_mass(events_tmm, Sel_Muon, Sel_Muon2, out, ds, tag)
        self.saved_MET(events_tmm, out, ds, tag)
        self.saved_drl1l2(events_tmm, Sel_Muon, Sel_Muon2, out, ds, tag)

        return events_tmm

    def analyse_tem_SS(self, events, out, ds):

        tag = 'tem_SS'
        # select tem_SS events: require 1 reco mu, 1 reco tau and 1 reco e + e and mu have the SS
        events_tem_SS = events[(ak.num(events.SelElectron) == 1) & (ak.num(events.SelMuon) == 1) & (ak.num(events.SelTau) == 1)] 
        events_tem_SS = events_tem_SS[ak.flatten(events_tem_SS.SelElectron.charge == events_tem_SS.SelMuon.charge)]

        out[f'sumw_3leptons_{tag}'][ds] += ak.sum(events_tem_SS.genWeight)
        out[f'n_ev_3leptons_{tag}'][ds] += len(events_tem_SS)

        # events should pass most efficient HLT (for now only 1: IsoMu24)
        events_tem_SS = events_tem_SS[events_tem_SS.HLT.IsoMu24]

        out[f'sumw_hlt_{tag}'][ds] += ak.sum(events_tem_SS.genWeight)
        out[f'n_ev_hlt_{tag}'][ds] += len(events_tem_SS)

        if len(events_tem_SS) == 0:
            return

        # select reco muon that match HLT IsoMu24
        Sel_Muon = self.select_lep1_IsoMu24(events_tem_SS, min_dr_cut=0.2)

        #removing non matching events
        events_tem_SS = events_tem_SS[(ak.num(Sel_Muon) == 1)]
        Sel_Muon = Sel_Muon[(ak.num(Sel_Muon) == 1)]

        out[f'sumw_l1sel_{tag}'][ds] += ak.sum(events_tem_SS.genWeight)
        out[f'n_ev_l1sel_{tag}'][ds] += len(events_tem_SS)

        # select reco Electron with dr(e,mu)>0.5 (in case there is more than 1 selected we choose the one with higher pt )
        Sel_Electron = self.select_lep2(events_tem_SS, Sel_Muon, type='electron', delta_r_cut = 0.5)

        #removing non matching events
        Sel_Muon = Sel_Muon[(ak.num(Sel_Electron) == 1)]
        events_tem_SS = events_tem_SS[(ak.num(Sel_Electron) == 1)]
        Sel_Electron = Sel_Electron[(ak.num(Sel_Electron) == 1)]

        out[f'sumw_l2sel_{tag}'][ds] += ak.sum(events_tem_SS.genWeight)
        out[f'n_ev_l2sel_{tag}'][ds] += len(events_tem_SS)

        # select reco Tau with dr(tau,mu)>0.5 and dr(tau,electron)>0.5 (in case there is more than 1 selected we choose the one with higher pt )
        Sel_Tau = self.select_lep3(events_tem_SS, Sel_Muon, Sel_Electron, type='tau', delta_r_cut = 0.5)

        #removing non matching events
        Sel_Muon = Sel_Muon[(ak.num(Sel_Tau) == 1)]
        Sel_Electron = Sel_Electron[(ak.num(Sel_Tau) == 1)]
        events_tem_SS = events_tem_SS[(ak.num(Sel_Tau) == 1)]
        Sel_Tau = Sel_Tau[(ak.num(Sel_Tau) == 1)]
        
        out[f'sumw_l3sel_{tag}'][ds] += ak.sum(events_tem_SS.genWeight)
        out[f'n_ev_l3sel_{tag}'][ds] += len(events_tem_SS)

        events_tem_SS, Sel_Tau, Sel_Electron, Sel_Muon = self.bjet_veto(events_tem_SS, Sel_Tau, Sel_Electron, Sel_Muon)
        out[f'sumw_bjetveto_{tag}'][ds] += ak.sum(events_tem_SS.genWeight)
        out[f'n_ev_bjetveto_{tag}'][ds] += len(events_tem_SS)

        events_tem_SS, Sel_Tau, Sel_Electron, Sel_Muon = self.charge_veto(events_tem_SS, Sel_Tau, Sel_Electron, Sel_Muon)
        out[f'sumw_chargeveto_{tag}'][ds] += ak.sum(events_tem_SS.genWeight)
        out[f'n_ev_chargeveto_{tag}'][ds] += len(events_tem_SS)

        events_tem_SS, Sel_Tau, Sel_Electron, Sel_Muon = self.met_veto(events_tem_SS, Sel_Tau, Sel_Electron, Sel_Muon)
        out[f'sumw_metselection_{tag}'][ds] += ak.sum(events_tem_SS.genWeight)
        out[f'n_ev_metselection_{tag}'][ds] += len(events_tem_SS)

        events_tem_SS, Sel_Electron, Sel_Tau, Sel_Muon = self.z_veto_tll(events_tem_SS, Sel_Electron, Sel_Tau, Sel_Muon)
        out[f'sumw_zveto_{tag}'][ds] += ak.sum(events_tem_SS.genWeight)
        out[f'n_ev_zveto_{tag}'][ds] += len(events_tem_SS)

        # Save histograms
        self.saved_leading_muon(events_tem_SS, Sel_Muon, out, ds, tag)
        self.saved_leading_tau(events_tem_SS, Sel_Tau, out, ds, tag)
        self.saved_leading_electron(events_tem_SS, Sel_Electron, out, ds, tag)
        self.saved_MET(events_tem_SS, out, ds, tag)
        self.saved_dilepton_mass(events_tem_SS, Sel_Tau, Sel_Electron, out, ds, tag)
        self.saved_drl1l2(events_tem_SS, Sel_Muon, Sel_Electron, out, ds, tag)

        return events_tem_SS

    def analyse_tem_OS(self, events, out, ds):

        tag = 'tem_OS'
        # select tem_OS events: require 1 reco mu, 1 reco tau and 1 reco e + e and mu have the OS
        events_tem_OS = events[(ak.num(events.SelElectron) == 1) & (ak.num(events.SelMuon) == 1) & (ak.num(events.SelTau) == 1)] 
        events_tem_OS = events_tem_OS[ak.flatten(events_tem_OS.SelElectron.charge != events_tem_OS.SelMuon.charge)]

        out[f'sumw_3leptons_{tag}'][ds] += ak.sum(events_tem_OS.genWeight)
        out[f'n_ev_3leptons_{tag}'][ds] += len(events_tem_OS)

        # events should pass most efficient HLT (for now only 1: IsoMu24)
        events_tem_OS = events_tem_OS[events_tem_OS.HLT.IsoMu24]

        out[f'sumw_hlt_{tag}'][ds] += ak.sum(events_tem_OS.genWeight)
        out[f'n_ev_hlt_{tag}'][ds] += len(events_tem_OS)

        if len(events_tem_OS) == 0:
            return

        # select reco muon that match HLT IsoMu24
        Sel_Muon = self.select_lep1_IsoMu24(events_tem_OS, min_dr_cut=0.2)

        #removing non matching events
        events_tem_OS = events_tem_OS[(ak.num(Sel_Muon) == 1)]
        Sel_Muon = Sel_Muon[(ak.num(Sel_Muon) == 1)]

        out[f'sumw_l1sel_{tag}'][ds] += ak.sum(events_tem_OS.genWeight)
        out[f'n_ev_l1sel_{tag}'][ds] += len(events_tem_OS)

        # select reco Electron with dr(e,mu)>0.5 (in case there is more than 1 selected we choose the one with higher pt )
        Sel_Electron = self.select_lep2(events_tem_OS, Sel_Muon, type='electron', delta_r_cut = 0.5)

        #removing non matching events
        Sel_Muon = Sel_Muon[(ak.num(Sel_Electron) == 1)]
        events_tem_OS = events_tem_OS[(ak.num(Sel_Electron) == 1)]
        Sel_Electron = Sel_Electron[(ak.num(Sel_Electron) == 1)]

        out[f'sumw_l2sel_{tag}'][ds] += ak.sum(events_tem_OS.genWeight)
        out[f'n_ev_l2sel_{tag}'][ds] += len(events_tem_OS)

        # select reco Tau with dr(tau,mu)>0.5 and dr(tau,electron)>0.5 (in case there is more than 1 selected we choose the one with higher pt )
        Sel_Tau = self.select_lep3(events_tem_OS, Sel_Muon, Sel_Electron, type='tau', delta_r_cut = 0.5)

        #removing non matching events
        Sel_Muon = Sel_Muon[(ak.num(Sel_Tau) == 1)]
        Sel_Electron = Sel_Electron[(ak.num(Sel_Tau) == 1)]
        events_tem_OS = events_tem_OS[(ak.num(Sel_Tau) == 1)]
        Sel_Tau = Sel_Tau[(ak.num(Sel_Tau) == 1)]
        
        out[f'sumw_l3sel_{tag}'][ds] += ak.sum(events_tem_OS.genWeight)
        out[f'n_ev_l3sel_{tag}'][ds] += len(events_tem_OS)

        events_tem_OS, Sel_Tau, Sel_Electron, Sel_Muon = self.bjet_veto(events_tem_OS, Sel_Tau, Sel_Electron, Sel_Muon)
        out[f'sumw_bjetveto_{tag}'][ds] += ak.sum(events_tem_OS.genWeight)
        out[f'n_ev_bjetveto_{tag}'][ds] += len(events_tem_OS)

        events_tem_OS, Sel_Tau, Sel_Electron, Sel_Muon = self.charge_veto(events_tem_OS, Sel_Tau, Sel_Electron, Sel_Muon)
        out[f'sumw_chargeveto_{tag}'][ds] += ak.sum(events_tem_OS.genWeight)
        out[f'n_ev_chargeveto_{tag}'][ds] += len(events_tem_OS)

        events_tem_OS, Sel_Tau, Sel_Electron, Sel_Muon = self.met_veto(events_tem_OS, Sel_Tau, Sel_Electron, Sel_Muon)
        out[f'sumw_metselection_{tag}'][ds] += ak.sum(events_tem_OS.genWeight)
        out[f'n_ev_metselection_{tag}'][ds] += len(events_tem_OS)

        events_tem_OS, Sel_Tau, Sel_Electron, Sel_Muon = self.z_veto_tll(events_tem_OS, Sel_Tau, Sel_Electron, Sel_Muon)
        out[f'sumw_zveto_{tag}'][ds] += ak.sum(events_tem_OS.genWeight)
        out[f'n_ev_zveto_{tag}'][ds] += len(events_tem_OS)

        # Save histograms
        self.saved_leading_muon(events_tem_OS, Sel_Muon, out, ds, tag)
        self.saved_leading_tau(events_tem_OS, Sel_Tau, out, ds, tag)
        self.saved_leading_electron(events_tem_OS, Sel_Electron, out, ds, tag)
        self.saved_MET(events_tem_OS, out, ds, tag)
        self.saved_dilepton_mass(events_tem_OS, Sel_Electron, Sel_Muon, out, ds, tag)
        self.saved_drl1l2(events_tem_OS, Sel_Muon, Sel_Electron, out, ds, tag)

        return events_tem_OS

    def analyse_tee(self, events, out, ds):

        tag = 'tee'
        # select tee events: require 2 reco e and 1 reco tau 
        events_tee = events[(ak.num(events.SelElectron) == 2) & (ak.num(events.SelTau) == 1)]

        out[f'sumw_3leptons_{tag}'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_3leptons_{tag}'][ds] += len(events_tee)

        # events should pass most efficient HLT (Ele32_WPTight_Gsf_L1DoubleEG)
        events_tee = events_tee[events_tee.HLT.Ele32_WPTight_Gsf_L1DoubleEG]

        out[f'sumw_hlt_{tag}'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_hlt_{tag}'][ds] += len(events_tee)

        if len(events_tee) == 0:
            return

        # select reco electron that match HLT Ele32_WPTight_Gsf_L1DoubleEG
        Sel_Electron = self.select_lep1_Ele32_WPTight_Gsf_L1DoubleEG(events_tee, min_dr_cut=0.2)

        #removing non matching events
        events_tee = events_tee[(ak.num(Sel_Electron) == 1)]
        Sel_Electron = Sel_Electron[(ak.num(Sel_Electron) == 1)]
    
        out[f'sumw_l1sel_{tag}'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_l1sel_{tag}'][ds] += len(events_tee)

        # select reco Electron2 with dr(e2,e)>0.5 (in case there is more than 1 selected we choose the one with higher pt )
        Sel_Electron2 = self.select_lep2(events_tee, Sel_Electron, type='electron', delta_r_cut = 0.5)

        #removing non matching events
        Sel_Electron = Sel_Electron[(ak.num(Sel_Electron2) == 1)]
        events_tee = events_tee[(ak.num(Sel_Electron2) == 1)]
        Sel_Electron2 = Sel_Electron2[(ak.num(Sel_Electron2) == 1)]

        out[f'sumw_l2sel_{tag}'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_l2sel_{tag}'][ds] += len(events_tee)

        # select reco Tau with dr(tau,e)>0.5 and dr(tau,e2)>0.5 (in case there is more than 1 selected we choose the one with higher pt )
        Sel_Tau = self.select_lep3(events_tee, Sel_Electron, Sel_Electron2, type='tau', delta_r_cut = 0.5)

        #removing non matching events
        Sel_Electron = Sel_Electron[(ak.num(Sel_Tau) == 1)]
        Sel_Electron2 = Sel_Electron2[(ak.num(Sel_Tau) == 1)]
        events_tee = events_tee[(ak.num(Sel_Tau) == 1)]
        Sel_Tau = Sel_Tau[(ak.num(Sel_Tau) == 1)]

        out[f'sumw_l3sel_{tag}'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_l3sel_{tag}'][ds] += len(events_tee)

        events_tee, Sel_Tau, Sel_Electron, Sel_Electron2 = self.bjet_veto(events_tee, Sel_Tau, Sel_Electron, Sel_Electron2)
        out[f'sumw_bjetveto_{tag}'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_bjetveto_{tag}'][ds] += len(events_tee)

        events_tee, Sel_Tau, Sel_Electron, Sel_Electron2 = self.charge_veto(events_tee, Sel_Tau, Sel_Electron, Sel_Electron2)
        out[f'sumw_chargeveto_{tag}'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_chargeveto_{tag}'][ds] += len(events_tee)

        events_tee, Sel_Tau, Sel_Electron, Sel_Electron2 = self.met_veto(events_tee, Sel_Tau, Sel_Electron, Sel_Electron2)
        out[f'sumw_metselection_{tag}'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_metselection_{tag}'][ds] += len(events_tee)

        events_tee, Sel_Tau, Sel_Electron, Sel_Electron2 = self.z_veto_tll(events_tee, Sel_Tau, Sel_Electron, Sel_Electron2)
        out[f'sumw_zveto_{tag}'][ds] += ak.sum(events_tee.genWeight)
        out[f'n_ev_zveto_{tag}'][ds] += len(events_tee)

        # Save histograms
        self.saved_leading_electron(events_tee, Sel_Electron, out, ds, tag)
        self.saved_subleading_electron(events_tee, Sel_Electron2, out, ds, tag)
        self.saved_leading_tau(events_tee, Sel_Tau, out, ds, tag)
        self.saved_dilepton_mass(events_tee, Sel_Electron, Sel_Electron2, out, ds, tag)
        self.saved_MET(events_tee, out, ds, tag)
        self.saved_drl1l2(events_tee, Sel_Electron, Sel_Electron2, out, ds, tag)

        return events_tee

    def analyse_tte(self, events, out, ds):

        tag = 'tte'
        # select tte events: require 2 reco Tau and 1 reco e 
        events_tte = events[(ak.num(events.SelElectron) == 1) & (ak.num(events.SelTau) == 2)]

        out[f'sumw_3leptons_{tag}'][ds] += ak.sum(events_tte.genWeight)
        out[f'n_ev_3leptons_{tag}'][ds] += len(events_tte)

        # events should pass most efficient HLT (DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg)
        events_tte = events_tte[events_tte.HLT.DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg]

        out[f'sumw_hlt_{tag}'][ds] += ak.sum(events_tte.genWeight)
        out[f'n_ev_hlt_{tag}'][ds] += len(events_tte)

        if len(events_tte) == 0:
            return

        # select reco tau1 and tau2 that match HLT DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg with dr(t2,t1)>0.5
        events_tte, Sel_Tau1, Sel_Tau2 = self.select_lep_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg(events_tte, min_dr_cut=0.5)

        out[f'sumw_l1sel_{tag}'][ds] += ak.sum(events_tte.genWeight)
        out[f'n_ev_l1sel_{tag}'][ds] += len(events_tte)

        out[f'sumw_l2sel_{tag}'][ds] += ak.sum(events_tte.genWeight)
        out[f'n_ev_l2sel_{tag}'][ds] += len(events_tte)

        # select reco electron with dr(tau1,e)>0.5 and dr(tau2,e)>0.5 (in case there is more than 1 selected we choose the one with higher pt )
        Sel_Electron = self.select_lep3(events_tte, Sel_Tau1, Sel_Tau2, type='electron', delta_r_cut = 0.5)

        #removing non matching events
        cut = (ak.num(Sel_Electron) == 1)
        Sel_Tau1 = Sel_Tau1[cut]
        Sel_Tau2 = Sel_Tau2[cut]
        events_tte = events_tte[cut]
        Sel_Electron = Sel_Electron[cut]

        out[f'sumw_l3sel_{tag}'][ds] += ak.sum(events_tte.genWeight)
        out[f'n_ev_l3sel_{tag}'][ds] += len(events_tte)

        events_tte, Sel_Tau1, Sel_Tau2, Sel_Electron = self.bjet_veto(events_tte, Sel_Tau1, Sel_Tau2, Sel_Electron)
        out[f'sumw_bjetveto_{tag}'][ds] += ak.sum(events_tte.genWeight)
        out[f'n_ev_bjetveto_{tag}'][ds] += len(events_tte)

        events_tte, Sel_Tau1, Sel_Tau2, Sel_Electron = self.charge_veto(events_tte, Sel_Tau1, Sel_Tau2, Sel_Electron)
        out[f'sumw_chargeveto_{tag}'][ds] += ak.sum(events_tte.genWeight)
        out[f'n_ev_chargeveto_{tag}'][ds] += len(events_tte)

        events_tte, Sel_Tau1, Sel_Tau2, Sel_Electron = self.met_veto(events_tte, Sel_Tau1, Sel_Tau2, Sel_Electron)
        out[f'sumw_metselection_{tag}'][ds] += ak.sum(events_tte.genWeight)
        out[f'n_ev_metselection_{tag}'][ds] += len(events_tte)

        events_tte, Sel_Tau1, Sel_Tau2, Sel_Electron = self.z_veto_ttl(events_tte, Sel_Tau1, Sel_Tau2, Sel_Electron)
        out[f'sumw_zveto_{tag}'][ds] += ak.sum(events_tte.genWeight)
        out[f'n_ev_zveto_{tag}'][ds] += len(events_tte)

        # Save histograms
        self.saved_leading_tau(events_tte, Sel_Tau1, out, ds, tag)
        self.saved_subleading_tau(events_tte, Sel_Tau2, out, ds, tag)
        self.saved_leading_electron(events_tte, Sel_Electron, out, ds, tag)
        self.saved_dilepton_mass_taul1_OS(events_tte, Sel_Electron, Sel_Tau1, Sel_Tau2, out, ds, tag)
        self.saved_MET(events_tte, out, ds, tag)
        self.saved_drl1l2(events_tte, Sel_Tau1, Sel_Tau2, out, ds, tag)

        return events_tte

    def analyse_ttt(self, events, out, ds):

        tag = 'ttt'
        # select ttt events: require 3 reco Tau 
        events_ttt = events[(ak.num(events.SelTau) == 3)]

        out[f'sumw_3leptons_{tag}'][ds] += ak.sum(events_ttt.genWeight)
        out[f'n_ev_3leptons_{tag}'][ds] += len(events_ttt)

        # events should pass most efficient HLT (oubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg)
        events_ttt = events_ttt[events_ttt.HLT.DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg]

        out[f'sumw_hlt_{tag}'][ds] += ak.sum(events_ttt.genWeight)
        out[f'n_ev_hlt_{tag}'][ds] += len(events_ttt)

        if len(events_ttt) == 0:
            return

        # select reco tau1 and tau2 that match HLT DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg with dr(t2,t1)>0.5
        events_ttt, Sel_Tau1, Sel_Tau2 = self.select_lep_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg(events_ttt, min_dr_cut=0.5)

        out[f'sumw_l1sel_{tag}'][ds] += ak.sum(events_ttt.genWeight)
        out[f'n_ev_l1sel_{tag}'][ds] += len(events_ttt)

        out[f'sumw_l2sel_{tag}'][ds] += ak.sum(events_ttt.genWeight)
        out[f'n_ev_l2sel_{tag}'][ds] += len(events_ttt)

        # select reco tau with dr(tau1,tau3)>0.5 and dr(tau2,tau3)>0.5 (in case there is more than 1 selected we choose the one with higher pt )
        Sel_Tau3 = self.select_lep3(events_ttt, Sel_Tau1, Sel_Tau2, type='tau', delta_r_cut = 0.5)

        #removing non matching events
        cut=(ak.num(Sel_Tau3) == 1)
        Sel_Tau1 = Sel_Tau1[cut]
        Sel_Tau2 = Sel_Tau2[cut]
        events_ttt = events_ttt[cut]
        Sel_Tau3 = Sel_Tau3[cut]

        out[f'sumw_l3sel_{tag}'][ds] += ak.sum(events_ttt.genWeight)
        out[f'n_ev_l3sel_{tag}'][ds] += len(events_ttt)

        events_ttt, Sel_Tau1, Sel_Tau2, Sel_Tau3 = self.bjet_veto(events_ttt, Sel_Tau1, Sel_Tau2, Sel_Tau3)
        out[f'sumw_bjetveto_{tag}'][ds] += ak.sum(events_ttt.genWeight)
        out[f'n_ev_bjetveto_{tag}'][ds] += len(events_ttt)

        events_ttt, Sel_Tau1, Sel_Tau2, Sel_Tau3 = self.charge_veto(events_ttt, Sel_Tau1, Sel_Tau2, Sel_Tau3)
        out[f'sumw_chargeveto_{tag}'][ds] += ak.sum(events_ttt.genWeight)
        out[f'n_ev_chargeveto_{tag}'][ds] += len(events_ttt)

        events_ttt, Sel_Tau1, Sel_Tau2, Sel_Tau3 = self.met_veto(events_ttt, Sel_Tau1, Sel_Tau2, Sel_Tau3)
        out[f'sumw_metselection_{tag}'][ds] += ak.sum(events_ttt.genWeight)
        out[f'n_ev_metselection_{tag}'][ds] += len(events_ttt)

        # Save histograms
        self.saved_leading_tau(events_ttt, Sel_Tau1, out, ds, tag)
        self.saved_subleading_tau(events_ttt, Sel_Tau2, out, ds, tag)
        self.saved_subsubleading_tau(events_ttt, Sel_Tau3, out, ds, tag)
        self.saved_MET(events_ttt, out, ds, tag)
        self.saved_drl1l2(events_ttt, Sel_Tau1, Sel_Tau2, out, ds, tag)

        return events_ttt



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

    def select_lep_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg(self, events, min_dr_cut=0.2):
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

    def bjet_veto(self, events, Lepton1, Lepton2, Lepton3):
        jets = events.Jet[events.Jet.pt > 25.]
        bjets = jets[jets.btagDeepFlavB > 0.2770]
        cut = (ak.num(bjets) == 0)
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
