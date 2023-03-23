import correctionlib
import awkward as ak
from coffea.lookup_tools import extractor
import os

def compute_sf_mu(Sel_Muon):
    # sf for muon following https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2018
    Sel_Muon = ak.unflatten(Sel_Muon, counts= 1)
    # sf for muon with 25<pt<120 GeV
    RECO_corr_medium_pT_mu = get_correction_mu( 'NUM_TrackerMuons_DEN_genTracks', '2018_UL', 'sf', Sel_Muon)
    ID_corr_medium_pT_mu = get_correction_mu('NUM_MediumID_DEN_TrackerMuons', '2018_UL', 'sf', Sel_Muon)
    ISO_corr_medium_pT_mu = get_correction_mu('NUM_LooseRelIso_DEN_MediumID', '2018_UL', 'sf', Sel_Muon)
    sf_medium_pT = ak.from_numpy(RECO_corr_medium_pT_mu*ID_corr_medium_pT_mu*ISO_corr_medium_pT_mu)

    # sf for muon with pt>120 GeV
    fRECO_path = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/muon/NUM_TrackerMuons_Highpt_abseta_p.json' # custom tab from twiki because sf are not in muon_Z.json
    evaluator_RECO = get_scales_fromjson(fRECO_path)

    RECO_corr_high_pT_mu = ak.to_numpy(evaluator_RECO["NUM_HighPtMuons/abseta_p_value"](abs(Sel_Muon.eta), Sel_Muon.p))
    ID_corr_high_pT_mu = get_correction_mu('NUM_HighPtID_DEN_TrackerMuons', '2018_UL', 'sf', Sel_Muon)
    ISO_corr_high_pT_mu = get_correction_mu('NUM_LooseRelIso_DEN_MediumID', '2018_UL', 'sf', Sel_Muon)
    sf_high_pT = ak.from_numpy(RECO_corr_high_pT_mu*ID_corr_high_pT_mu*ISO_corr_high_pT_mu)

    sf = ak.flatten(ak.concatenate([ak.mask(sf_medium_pT, Sel_Muon.pt <= 120), ak.mask(sf_high_pT, Sel_Muon.pt > 120)], axis=1))

    return sf[~ak.is_none(sf)]

def get_trigger_correction_mu(Sel_Muon):
    Sel_Muon = ak.unflatten(Sel_Muon, counts= 1)

    # correction for Isomu24 trigger

    # for muon with 25<pt<120 GeV
    fTrigger_path = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/muon/Efficiencies_muon_generalTracks_Z_Run2018_UL_SingleMuonTriggers.root' 
    # file from https://gitlab.cern.ch/cms-muonPOG/muonefficiencies/-/blob/master/Run2/UL/2018/2018_trigger/Efficiencies_muon_generalTracks_Z_Run2018_UL_SingleMuonTriggers_schemaV2.json
    evaluator_Trigger = get_scales_fromjson(fTrigger_path)
    #Trigger_efficiency_corr_medium_pT = evaluator_Trigger["NUM_IsoMu24_DEN_CutBasedIdMedium_and_PFIsoMedium_eta_pt"](Sel_Muon.eta[Sel_Muon.pt <= 120], Sel_Muon.pt[Sel_Muon.pt <= 120])
    Trigger_efficiency_corr_medium_pT = evaluator_Trigger["NUM_IsoMu24_DEN_CutBasedIdMedium_and_PFIsoMedium_eta_pt"](ak.mask(Sel_Muon, Sel_Muon.pt <= 120).eta, ak.mask(Sel_Muon, Sel_Muon.pt <= 120).pt) 

    # for muon with pt>120 GeV
    fTrigger_path = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/muon/Highpt_muon_trigger_eff.json' 
    # https://gitlab.cern.ch/cms-muonPOG/muonefficiencies/-/blob/master/Run2/UL/2018/2018_trigger/Efficiencies_muon_generalTracks_Z_Run2018_UL_SingleMuonTriggers_schemaV2.json
    evaluator_Trigger = get_scales_fromjson(fTrigger_path)
    #Trigger_efficiency_corr_high_pT = evaluator_Trigger["NUM_HighPtMuons/abseta_p_value"](Sel_Muon.eta[Sel_Muon.pt > 120], Sel_Muon.pt[Sel_Muon.pt > 120])
    Trigger_efficiency_corr_high_pT = evaluator_Trigger["NUM_HighPtMuons/abseta_p_value"](ak.mask(Sel_Muon, Sel_Muon.pt > 120).eta, ak.mask(Sel_Muon, Sel_Muon.pt > 120).pt)

    Trigger_efficiency_corr = ak.flatten(ak.concatenate([Trigger_efficiency_corr_medium_pT, Trigger_efficiency_corr_high_pT], axis=1))

    return Trigger_efficiency_corr[~ak.is_none(Trigger_efficiency_corr)]

def compute_sf_e(Sel_Electron):
    # sf for electron following https://twiki.cern.ch/twiki/bin/view/CMS/EgammaUL2016To2018#SFs_for_Electrons_UL_2018
    RECO_corr_e = get_correction_e('UL-Electron-ID-SF', '2018', 'RecoAbove20','sf', Sel_Electron)
    #ID_corr_e = self.get_correction_e('UL-Electron-ID-SF', '2018', 'Medium','sf', Sel_Electron)
    MVA_corr_e = get_correction_e('UL-Electron-ID-SF', '2018', 'wp90iso','sf', Sel_Electron)
    #sf = ak.from_numpy(RECO_corr_e*ID_corr_e*MVA_corr_e)
    sf = ak.from_numpy(RECO_corr_e*MVA_corr_e)

    return sf

def get_trigger_correction_e(Sel_Electron):
    # sf for e trigger  from previous analysis: https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgHLTScaleFactorMeasurements
    fTrigger_path = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/electron/Riccardo_egammaTriggerEfficiency_2018_20200422.root' 
    evaluator_Trigger = get_scales_fromjson(fTrigger_path)
    Trigger_efficiency_corr = evaluator_Trigger["EGamma_SF2D"](Sel_Electron.eta, Sel_Electron.pt)

    return Trigger_efficiency_corr

def compute_sf_tau(Sel_Tau):
    # sf for tau with DeepTau2017v2p1 following https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
    DeepTauVSe_corr = get_correction_tau("DeepTau2017v2p1VSe", 'nom', Sel_Tau)
    DeepTauVSmu_corr = get_correction_tau("DeepTau2017v2p1VSmu", 'nom', Sel_Tau)
    DeepTauVSjet_corr = get_correction_tau("DeepTau2017v2p1VSjet", 'nom', Sel_Tau)
    NRJscale_corr = get_correction_tau("tau_energy_scale",'nom', Sel_Tau)
    sf = ak.from_numpy(DeepTauVSjet_corr*DeepTauVSe_corr*DeepTauVSmu_corr*NRJscale_corr)

    return ak.flatten(sf)

def compute_sf_tau_e(Sel_Tau):
    # sf for tau with DeepTau2017v2p1 following https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
    NRJscale_corr = get_correction_tau("tau_energy_scale",'nom', Sel_Tau)
    sf = ak.from_numpy(NRJscale_corr)

    return sf

def get_trigger_correction_tau(Sel_Tau):
    # sf for tau trigger following https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
    trigger_corr = get_correction_tau("tau_trigger",'nom', Sel_Tau)
    sf = ak.from_numpy(trigger_corr)

    return sf

def get_scales_fromjson(json_file):
    ext = extractor()
    ext.add_weight_sets([f"* * {json_file}"])
    ext.finalize()
    evaluator = ext.make_evaluator()
    return evaluator

def get_correction_mu(corr_name, year, corr_type, lepton):
    f_path_mu = '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2018_UL/muon_Z.json.gz'
    ceval = correctionlib.CorrectionSet.from_file(f_path_mu)
    corr = ceval[corr_name].evaluate(year, ak.to_numpy(abs(lepton.eta)), ak.to_numpy(lepton.pt), corr_type)
    return corr

def get_correction_e(corr_name, year, WP, corr_type, lepton):
    f_path_e = '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2018_UL/electron.json.gz'
    ceval = correctionlib.CorrectionSet.from_file(f_path_e)
    corr = ceval[corr_name].evaluate(year, corr_type, WP, ak.to_numpy(lepton.eta), ak.to_numpy(lepton.pt))
    return corr

def get_correction_tau(corr_name, syst, lepton):
    f_path_tau = '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/TAU/2018_UL/tau.json.gz'
    ceval = correctionlib.CorrectionSet.from_file(f_path_tau)
    if corr_name == "DeepTau2017v2p1VSe":
        corr = ceval[corr_name].evaluate(ak.to_numpy(lepton.eta), ak.to_numpy(lepton.genPartFlav), 'VLoose', syst)
    if corr_name =="DeepTau2017v2p1VSmu":
        corr = ceval[corr_name].evaluate(ak.to_numpy(lepton.eta), ak.to_numpy(lepton.genPartFlav), 'Tight', syst)
    if corr_name == "DeepTau2017v2p1VSjet":
        corr = ceval[corr_name].evaluate(ak.to_numpy(lepton.pt), ak.to_numpy(lepton.decayMode), ak.to_numpy(lepton.genPartFlav), 'Medium', syst, "pt")
    if corr_name == "tau_trigger":
        corr = ceval[corr_name].evaluate(ak.to_numpy(lepton.pt), ak.to_numpy(lepton.decayMode), 'ditau', 'Medium', 'sf', syst)
    if corr_name == "tau_energy_scale":
        corr = ceval[corr_name].evaluate(ak.to_numpy(lepton.pt), ak.to_numpy(lepton.eta), ak.to_numpy(lepton.decayMode), ak.to_numpy(lepton.genPartFlav), "DeepTau2017v2p1", syst)
    return corr

def get_pileup_correction(events, syst):
    f_path = '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/LUM/2018_UL/puWeights.json.gz'
    ceval = correctionlib.CorrectionSet.from_file(f_path)
    corr = ceval['Collisions18_UltraLegacy_goldenJSON'].evaluate(ak.to_numpy(events.Pileup.nTrueInt), syst)
    return corr