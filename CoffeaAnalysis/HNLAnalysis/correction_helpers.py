import correctionlib
import awkward as ak
from coffea.lookup_tools import extractor
import os
import numpy as np

def get_scales_fromjson(json_file):
    ext = extractor()
    ext.add_weight_sets([f"* * {json_file}"])
    ext.finalize()
    evaluator = ext.make_evaluator()
    return evaluator

def get_correction_mu(corr_name, year, corr_type, lepton):
    #load correction from central repo
    f_path_mu = '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2018_UL/muon_Z.json.gz'
    ceval = correctionlib.CorrectionSet.from_file(f_path_mu)
    corr = ceval[corr_name].evaluate(year, ak.to_numpy(abs(lepton.eta)), ak.to_numpy(lepton.pt), corr_type)
    return corr

def get_correction_e(corr_name, year, WP, corr_type, lepton):
    #load correction from central repo
    f_path_e = '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2018_UL/electron.json.gz'
    ceval = correctionlib.CorrectionSet.from_file(f_path_e)
    corr = ceval[corr_name].evaluate(year, corr_type, WP, ak.to_numpy(lepton.eta), ak.to_numpy(lepton.pt))
    return corr

def get_correction_tau(corr_name, syst, lepton):
    #load correction from central repo
    f_path_tau = '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/TAU/2018_UL/tau.json.gz'
    ceval = correctionlib.CorrectionSet.from_file(f_path_tau)
    #extract corrections
    if corr_name == "DeepTau2017v2p1VSe":
        corr = ceval[corr_name].evaluate(ak.to_numpy(lepton.eta), ak.to_numpy(lepton.genPartFlav), 'VLoose', syst)
    if corr_name =="DeepTau2017v2p1VSmu":
        corr = ceval[corr_name].evaluate(ak.to_numpy(lepton.eta), ak.to_numpy(lepton.genPartFlav), 'Tight', syst)
    if corr_name == "DeepTau2017v2p1VSjet":
        corr = ceval[corr_name].evaluate(ak.to_numpy(lepton.pt), ak.to_numpy(lepton.decayMode), ak.to_numpy(lepton.genPartFlav), 'Medium', syst, "pt")
    if corr_name == "tau_trigger_ditau":
        corr = ceval["tau_trigger"].evaluate(ak.to_numpy(lepton.ptcorr), ak.to_numpy(lepton.decayMode), 'ditau', 'Medium', 'sf', syst)
    if corr_name == "tau_energy_scale":
        # until correctionlib handles jagged data natively we have to flatten and unflatten
        flat_lep, nlep = ak.flatten(lepton), ak.num(lepton)
        corr = ceval[corr_name].evaluate(ak.to_numpy(flat_lep.pt), ak.to_numpy(flat_lep.eta), ak.to_numpy(flat_lep.decayMode), ak.to_numpy(flat_lep.genPartFlav), "DeepTau2017v2p1", syst)
        return ak.unflatten(corr, nlep)
    return corr

def compute_sf_tau(Sel_Tau):
    # sf for genuine tau (https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2)
    f_path = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/tau/NUM_Tau_Medium_DecayMode.json' # custom tab for tau sf with DeepTau2017v2p5VSJet
    evaluator = get_scales_fromjson(f_path)

    nTrueHadronicTau = Sel_Tau.genPartFlav == 5
    nFakeHadronicTau = Sel_Tau.genPartFlav != 5
    tau_sf_TrueHadronicTau = ak.to_numpy(evaluator["DeepTau2017v2p5/DecayMode_value"](Sel_Tau.decayMode[Sel_Tau.genPartFlav == 5]))
    tau_sf_FakeHadronicTau = np.ones(len(Sel_Tau[Sel_Tau.genPartFlav != 5])) # sf set to 1. for Fake Taus

    tau_sf = ak.concatenate([ak.unflatten(tau_sf_TrueHadronicTau, nTrueHadronicTau*1), ak.unflatten(tau_sf_FakeHadronicTau, nFakeHadronicTau*1)], axis=1)

    return ak.flatten(tau_sf)

def compute_sf_mu(Sel_Muon):
    # sf for muon (https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2018)
    nlepBelow20 = Sel_Muon.pt < 20
    nlepBetween20and120 = (Sel_Muon.pt >= 20) & (Sel_Muon.pt <= 120)
    nlepAbove120 = Sel_Muon.pt > 120

    # sf for muon with pt<20 GeV
    f_path =  f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/muon/Efficiency_muon_generalTracks_Run2018_UL_trackerMuon.json' # custom tab from twiki because RECO sf in muon_Z.json are for Muon.pt > 15
    RECOevaluator_Below20 = get_scales_fromjson(f_path)
    f_path =  f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/muon/Efficiency_muon_trackerMuon_Run2018_UL_ID.json' # custom tab from twiki because ID sf in muon_Z.json are for Muon.pt > 15
    IDevaluator_Below20 = get_scales_fromjson(f_path)
    RECOsf_Below20 = ak.to_numpy(RECOevaluator_Below20["NUM_TrackerMuons_DEN_genTracks/abseta_pt_value"](abs(Sel_Muon.eta[Sel_Muon.pt < 20]), Sel_Muon.pt[Sel_Muon.pt < 20])) 
    IDsf_Below20 = ak.to_numpy(IDevaluator_Below20["NUM_MediumID_DEN_TrackerMuons/abseta_pt_value"](abs(Sel_Muon.eta[Sel_Muon.pt < 20]), Sel_Muon.pt[Sel_Muon.pt < 20])) 
    sf_Below20 = ak.from_numpy(RECOsf_Below20*IDsf_Below20)

    # sf for muon with 20<pt<120 GeV (from central repo)
    RECOsf_Between20and120 = get_correction_mu('NUM_TrackerMuons_DEN_genTracks', '2018_UL', 'sf', Sel_Muon[(Sel_Muon.pt >= 20) & (Sel_Muon.pt <= 120)])
    IDsf_Between20and120 = get_correction_mu('NUM_MediumID_DEN_TrackerMuons', '2018_UL', 'sf', Sel_Muon[(Sel_Muon.pt >= 20) & (Sel_Muon.pt <= 120)])
    ISOsf_Between20and120 = get_correction_mu('NUM_LooseRelIso_DEN_MediumID', '2018_UL', 'sf', Sel_Muon[(Sel_Muon.pt >= 20) & (Sel_Muon.pt <= 120)])
    sf_Between20and120 = ak.from_numpy(RECOsf_Between20and120*IDsf_Between20and120*ISOsf_Between20and120)

    # sf for muon with pt>120 GeV (ID and ISO same as muon with 20<pt<120 GeV)
    f_path = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/muon/NUM_TrackerMuons_Highpt_abseta_p.json' # custom tab from twiki because sf are not in muon_Z.json
    RECOevaluator_Above120 = get_scales_fromjson(f_path)
    RECOsf_Above120 = ak.to_numpy(RECOevaluator_Above120["NUM_HighPtMuons/abseta_p_value"](abs(Sel_Muon.eta[Sel_Muon.pt > 120]), Sel_Muon.p[Sel_Muon.pt > 120]))
    IDsf_Above120 = get_correction_mu('NUM_HighPtID_DEN_TrackerMuons', '2018_UL', 'sf', Sel_Muon[Sel_Muon.pt > 120])
    ISOsf_Above120 = get_correction_mu('NUM_LooseRelIso_DEN_MediumID', '2018_UL', 'sf', Sel_Muon[Sel_Muon.pt > 120])
    sf_Above120 = ak.from_numpy(RECOsf_Above120*IDsf_Above120*ISOsf_Above120)

    sf_mu = ak.concatenate([ak.unflatten(sf_Between20and120, nlepBetween20and120*1), ak.unflatten(sf_Above120, nlepAbove120*1), ak.unflatten(sf_Below20, nlepBelow20*1)], axis=1)

    return ak.flatten(sf_mu)

def compute_sf_e(Sel_Electron):
    ID_sf = get_correction_e('UL-Electron-ID-SF', '2018', 'wp90noiso','sf', Sel_Electron)

    return ID_sf

def get_trigger_correction_mu(Sel_Muon, HLTname = 'IsoMu24'):
    Sel_Muon = ak.unflatten(Sel_Muon, counts= 1)

    if HLTname == 'IsoMu24':
        # correction for IsoMu24 trigger

        # for muon with 25<pt<120 GeV
        fTrigger_path = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/muon/Efficiencies_muon_generalTracks_Z_Run2018_UL_SingleMuonTriggers.root' 
        # file from https://gitlab.cern.ch/cms-muonPOG/muonefficiencies/-/blob/master/Run2/UL/2018/2018_trigger/Efficiencies_muon_generalTracks_Z_Run2018_UL_SingleMuonTriggers_schemaV2.json
        evaluator_Trigger = get_scales_fromjson(fTrigger_path)
        Trigger_efficiency_corr_medium_pT = evaluator_Trigger["NUM_IsoMu24_DEN_CutBasedIdMedium_and_PFIsoMedium_eta_pt"](ak.mask(Sel_Muon, Sel_Muon.pt <= 120).eta, ak.mask(Sel_Muon, Sel_Muon.pt <= 120).pt) 

        # for muon with pt>120 GeV
        fTrigger_path = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/muon/Highpt_muon_trigger_eff.json' 
        # file from https://gitlab.cern.ch/cms-muonPOG/muonefficiencies/-/blob/master/Run2/UL/2018/2018_trigger/Efficiencies_muon_generalTracks_Z_Run2018_UL_SingleMuonTriggers_schemaV2.json
        evaluator_Trigger = get_scales_fromjson(fTrigger_path)
        Trigger_efficiency_corr_high_pT = evaluator_Trigger["NUM_HighPtMuons/abseta_p_value"](ak.mask(Sel_Muon, Sel_Muon.pt > 120).eta, ak.mask(Sel_Muon, Sel_Muon.pt > 120).pt)

        Trigger_efficiency_corr = ak.flatten(ak.concatenate([Trigger_efficiency_corr_medium_pT, Trigger_efficiency_corr_high_pT], axis=1))
        Trigger_efficiency_corr = Trigger_efficiency_corr[~ak.is_none(Trigger_efficiency_corr)]

    return Trigger_efficiency_corr

def get_trigger_correction_e(Sel_Electron, HLTname = 'Ele32'):
    if HLTname == 'Ele32':
        # sf for e trigger from previous analysis: https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgHLTScaleFactorMeasurements
        fTrigger_path = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/electron/Riccardo_egammaTriggerEfficiency_2018_20200422.root' 
        evaluator_Trigger = get_scales_fromjson(fTrigger_path)
        Trigger_efficiency_corr = evaluator_Trigger["EGamma_SF2D"](Sel_Electron.eta, Sel_Electron.pt)

    return Trigger_efficiency_corr

def get_trigger_correction_tau(Sel_Tau, HLTname = 'diTau'):
    if HLTname == 'diTau':
        # sf for tau trigger following https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
        trigger_corr = get_correction_tau("tau_trigger_ditau",'nom', Sel_Tau)
        sf = ak.from_numpy(trigger_corr)

    return sf

def compute_tau_e_corr(Sel_Tau):
    # energy correction for tau with DeepTau2017v2p1 following https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
    NRJscale_corr = get_correction_tau("tau_energy_scale",'nom', Sel_Tau)
    return NRJscale_corr

def get_pileup_correction(events, syst):
    f_path = '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/LUM/2018_UL/puWeights.json.gz'
    ceval = correctionlib.CorrectionSet.from_file(f_path)
    corr = ceval['Collisions18_UltraLegacy_goldenJSON'].evaluate(ak.to_numpy(events.Pileup.nTrueInt), syst)
    return corr