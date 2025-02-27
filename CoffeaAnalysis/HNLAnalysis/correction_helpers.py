import correctionlib
import awkward as ak
from coffea.lookup_tools import extractor
import os
import numpy as np
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory, CorrectedMETFactory

#global parameters
Area_dir = {
    '2016_HIPM': '2016preVFP_UL',
    '2016': '2016postVFP_UL',
    '2017': '2017_UL',
    '2018': '2018_UL'
}
Area_ref = {
    '2016_HIPM': '2016_preVFP',
    '2016': '2016_postVFP',
    '2017': '2017',
    '2018': '2018'
}

def get_scales_fromjson(json_file):
    ext = extractor()
    ext.add_weight_sets([f"* * {json_file}"])
    ext.finalize()
    evaluator = ext.make_evaluator()
    return evaluator

def get_correction_central(corr, period):
    POG_path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/"
    #load correction from central repo
    if corr == 'muon':
        f_path = os.path.join(POG_path, 'MUO/' + Area_dir[period] + '/muon_Z.json.gz')
    if corr == 'tau':
        f_path = os.path.join(POG_path, 'TAU/' + Area_dir[period] + '/tau.json.gz')
    if corr == 'electron':
        f_path = os.path.join(POG_path, 'EGM/' + Area_dir[period] + '/electron.json.gz')
    if corr == 'pileup':
        f_path = os.path.join(POG_path, 'LUM/' + Area_dir[period] + '/puWeights.json.gz')
    if corr == 'btag':
        f_path = os.path.join(POG_path, 'BTV/' + Area_dir[period] + '/btagging.json.gz')
    ceval = correctionlib.CorrectionSet.from_file(f_path)
    return ceval

def compute_sf_tau(Sel_Tau, events, name, period, DeepTauVersion):
    # sf for tau (https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2)
    if DeepTauVersion == "DeepTau2018v2p5":
        #load correction from custom json
        f_path = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/tau/data/{period}/tau_DeepTau2018v2p5_UL{Area_ref[period]}.json'
        ceval = correctionlib.CorrectionSet.from_file(f_path)
    if DeepTauVersion == "DeepTau2017v2p1":
        #load correction from central repo
        ceval = get_correction_central('tau', period)

    #Corrections to be applied to genuine electrons misidentified as taus
    sf_Vse = weightcorr_TauID_genuineElectron(events, Sel_Tau, name, ceval, period, DeepTauVersion)
    #Corrections to be applied to genuine muons misidentified as taus
    sf_Vsmu = weightcorr_TauID_genuineMuon(events, Sel_Tau, name, ceval, period, DeepTauVersion)
    #Corrections to be applied to genuine Taus
    sf_VsJet = weightcorr_TauID_genuineTau(events, Sel_Tau, name, ceval, period, DeepTauVersion)
    tau_sf = sf_Vse*sf_Vsmu*sf_VsJet
    events[f'weightcorr_{name}_TauID_Total_Central'] = tau_sf
    return tau_sf

def compute_sf_mu(Sel_Muon, events, name, period):
    # sf for muon (https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2018)
    ceval = get_correction_central('muon', period)
    #Corrections due to RECO efficiencies
    sf_RECO = weightcorr_MuID_RECO(events, Sel_Muon, name, ceval, period)
    #Corrections due to ID efficiencies
    sf_ID = weightcorr_MuID_MediumID(events, Sel_Muon, name, ceval, period)
    #Corrections due to ISO efficiencies
    sf_ISO = weightcorr_MuID_LooseISO(events, Sel_Muon, name, ceval, period)
    mu_sf = sf_RECO*sf_ID*sf_ISO
    events[f'weightcorr_{name}_MuID_Total_Central'] = mu_sf
    return mu_sf

def compute_sf_e(Sel_Electron, events, name, period):
    # sf for electrons (https://twiki.cern.ch/twiki/bin/view/CMS/EgammaUL2016To2018)
    ceval = get_correction_central('electron', period)
    #Corrections due to RECO efficiencies
    sf_RECO = weightcorr_EleID_RECO(events, Sel_Electron, name, period)
    #Corrections due to ID efficiencies
    sf_ID = weightcorr_EleID_wp90noiso(events, Sel_Electron, name, ceval, period)
    e_sf = sf_RECO*sf_ID
    events[f'weightcorr_{name}_EleID_Total_Central'] = e_sf
    return e_sf

def get_trigger_correction_mu(Sel_Muon, events, name, period):
    HLT_name = {
        '2018': 'NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight',
        '2017': 'NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight',
        '2016_HIPM': 'NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight',
        '2016': 'NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight',
    }
    ev_pt = (Sel_Muon.pt)
    if period == '2017': #IsoMu27
        ev_pt = np.where(ev_pt<29, 29, ev_pt)
    else: #IsoMu24
        ev_pt = np.where(ev_pt<26, 26, ev_pt)

    ceval = get_correction_central('muon', period)
    TrgSF = ceval[HLT_name[period]].evaluate(np.abs(Sel_Muon.eta), ev_pt, "nominal")
    events[f'weightcorr_{name}_TrgSF_singleMu_Total_Central'] = TrgSF
    events[f'weightcorr_{name}_TrgSF_singleMu_syst_Up_rel'] =    (TrgSF + ceval[HLT_name[period]].evaluate(np.abs(Sel_Muon.eta), ev_pt, "syst"))/TrgSF
    events[f'weightcorr_{name}_TrgSF_singleMu_syst_Down_rel'] =  (TrgSF - ceval[HLT_name[period]].evaluate(np.abs(Sel_Muon.eta), ev_pt, "syst"))/TrgSF
    events[f'weightcorr_{name}_TrgSF_singleMu_stat_{period}_Up_rel'] =    (TrgSF + ceval[HLT_name[period]].evaluate(np.abs(Sel_Muon.eta), ev_pt, "stat"))/TrgSF
    events[f'weightcorr_{name}_TrgSF_singleMu_stat_{period}_Down_rel'] =  (TrgSF - ceval[HLT_name[period]].evaluate(np.abs(Sel_Muon.eta), ev_pt, "stat"))/TrgSF
    return TrgSF

def get_trigger_correction_e(Sel_Electron, events, name, period):
    HLT_name = {
        '2018': 'Ele32',
        '2017': 'Ele32',
        '2016': 'Ele25',
        '2016_HIPM': 'Ele25',
    }
    # sf for e trigger from previous analysis: https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgHLTScaleFactorMeasurements
    fTrigger_path = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/electron/data/{period}/sf_el_{period}_HLT{HLT_name[period]}_witherr.root' 
    evaluator_Trigger = get_scales_fromjson(fTrigger_path)
    e_pt = ak.to_numpy(Sel_Electron.pt)
    e_pt = np.where(Sel_Electron.pt<20, 20, Sel_Electron.pt)
    Trigger_efficiency_corr = evaluator_Trigger["SF2D"](e_pt, Sel_Electron.eta)
    events[f'weightcorr_{name}_TrgSF_singleEle_stat_Total_Central'] = Trigger_efficiency_corr
    events[f'weightcorr_{name}_TrgSF_singleEle_stat_{period}_Up_rel'] =   (Trigger_efficiency_corr + evaluator_Trigger["SF2D_err"](e_pt, Sel_Electron.eta))/Trigger_efficiency_corr
    events[f'weightcorr_{name}_TrgSF_singleEle_stat_{period}_Down_rel'] = (Trigger_efficiency_corr - evaluator_Trigger["SF2D_err"](e_pt, Sel_Electron.eta))/Trigger_efficiency_corr
    #syst unc: flat 3% correlated directly in the datacard creation code
    return Trigger_efficiency_corr

def compute_electron_ES_corr(Sel_Electron):
    # energy scale correction for electron (info ntuples)
    ES_up = (Sel_Electron.energy + Sel_Electron.dEscaleUp)/Sel_Electron.energy
    ES_down = (Sel_Electron.energy + Sel_Electron.dEscaleDown)/Sel_Electron.energy
    return ES_up, ES_down

def compute_electron_ER_corr(Sel_Electron):
    # energy smearing correction for electron (info ntuples)
    ES_up = (Sel_Electron.energy + Sel_Electron.dEsigmaUp)/Sel_Electron.energy
    ES_down = (Sel_Electron.energy + Sel_Electron.dEsigmaDown)/Sel_Electron.energy
    return ES_up, ES_down

def get_trigger_correction_tau(Sel_Tau, events, name, period):
    #load correction from central repo
    ceval = get_correction_central('tau', period)
    sf = ak.to_numpy(ceval["tau_trigger"].evaluate(ak.to_numpy(Sel_Tau.pt), ak.to_numpy(Sel_Tau.decayMode), 'ditau', 'Medium', 'sf', 'nom'))
    sf_Up = ak.to_numpy(ceval["tau_trigger"].evaluate(ak.to_numpy(Sel_Tau.pt), ak.to_numpy(Sel_Tau.decayMode), 'ditau', 'Medium', 'sf', 'up'))
    sf_Down = ak.to_numpy(ceval["tau_trigger"].evaluate(ak.to_numpy(Sel_Tau.pt), ak.to_numpy(Sel_Tau.decayMode), 'ditau', 'Medium', 'sf', 'down'))

    DecayModes = ['0', '1', '3Prong']
    for dm in DecayModes:
        list_up = []
        list_down = []
        for decayM in DecayModes:
            if (decayM == '0') or (decayM == '1'):
                Sel_Tau_dm = Sel_Tau.decayMode == int(decayM)
            if (decayM == '3Prong'):
                Sel_Tau_dm = (Sel_Tau.decayMode == 10)|(Sel_Tau.decayMode == 11)

            if decayM == dm:
                list_up.append(ak.unflatten(sf_Up[Sel_Tau_dm]/sf[Sel_Tau_dm], Sel_Tau_dm*1))
                list_down.append(ak.unflatten(sf_Down[Sel_Tau_dm]/sf[Sel_Tau_dm], Sel_Tau_dm*1))
            else:
                list_up.append(ak.unflatten(np.ones(len(Sel_Tau_dm))[Sel_Tau_dm], Sel_Tau_dm*1))
                list_down.append(ak.unflatten(np.ones(len(Sel_Tau_dm))[Sel_Tau_dm], Sel_Tau_dm*1))
        events[f'weightcorr_{name}_TrgSF_ditau_DM{dm}_{period}_Up_rel'] = ak.concatenate(list_up, axis=1)
        events[f'weightcorr_{name}_TrgSF_ditau_DM{dm}_{period}_Down_rel'] =ak.concatenate(list_down, axis=1)

    events[f'weightcorr_{name}_TrgSF_ditau_DM{dm}_{period}_Total_Up'] = sf_Up
    events[f'weightcorr_{name}_TrgSF_ditau_DM{dm}_{period}_Total_Down'] = sf_Down
    events[f'weightcorr_{name}_TrgSF_ditau_DM{dm}_{period}_Central'] = sf
    return events, sf

def compute_tau_e_corr(Sel_Tau, period):
    # energy correction for tau with DeepTau2017v2p1 following https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
    #load correction from central repo
    ceval = get_correction_central('tau', period)
    # until correctionlib handles jagged data natively we have to flatten and unflatten
    flat_lep, nlep = ak.flatten(Sel_Tau), ak.num(Sel_Tau)

    corr = ceval["tau_energy_scale"].evaluate(flat_lep.pt, flat_lep.eta, flat_lep.decayMode, flat_lep.genPartFlav, "DeepTau2017v2p1", 'nom')
    corr_up = ceval["tau_energy_scale"].evaluate(flat_lep.pt, flat_lep.eta, flat_lep.decayMode, flat_lep.genPartFlav, "DeepTau2017v2p1", 'up')
    corr_down = ceval["tau_energy_scale"].evaluate(flat_lep.pt, flat_lep.eta, flat_lep.decayMode, flat_lep.genPartFlav, "DeepTau2017v2p1", 'down')
    NRJscale_corr = ak.unflatten(corr, nlep)
    NRJscale_corr_up = ak.unflatten(corr_up, nlep)
    NRJscale_corr_down = ak.unflatten(corr_down, nlep)

    return NRJscale_corr, NRJscale_corr_up, NRJscale_corr_down

def get_pileup_correction(events, period, save_updown=True):
    PU_key = {
        '2016_HIPM': 'Collisions16_UltraLegacy_goldenJSON',
        '2016': 'Collisions16_UltraLegacy_goldenJSON',
        '2017': 'Collisions17_UltraLegacy_goldenJSON',
        '2018': 'Collisions18_UltraLegacy_goldenJSON'
    }
    ceval = get_correction_central('pileup', period)
    corr = ceval[PU_key[period]].evaluate(events.Pileup.nTrueInt, 'nominal')
    corr_up = ceval[PU_key[period]].evaluate(events.Pileup.nTrueInt, 'up')
    corr_down = ceval[PU_key[period]].evaluate(events.Pileup.nTrueInt, 'down')

    if save_updown:
        events[f'weightcorr_PileUp_Total_Central'] = corr
        events[f'weightcorr_PileUp_Up_rel'] = corr_up/corr
        events[f'weightcorr_PileUp_Down_rel'] = corr_down/corr
    return corr, corr_up, corr_down

def compute_sf_L1PreFiring(events):
    sf_L1PreFiring = events['L1PreFiringWeight']['Nom']
    events['weightcorr_L1PreFiring_Total_Central'] = sf_L1PreFiring
    #events['weightcorr_L1PreFiring_Total_Up'] = ak.to_numpy(events['L1PreFiringWeight']['Up'])
    #events['weightcorr_L1PreFiring_Total_Down'] = ak.to_numpy(events['L1PreFiringWeight']['Dn'])
    events['weightcorr_L1PreFiring_ECAL_Central_rel'] = events['L1PreFiringWeight']['ECAL_Nom']/sf_L1PreFiring
    events['weightcorr_L1PreFiring_ECALUp_rel'] = events['L1PreFiringWeight']['ECAL_Up']/sf_L1PreFiring
    events['weightcorr_L1PreFiring_ECALDown_rel'] = events['L1PreFiringWeight']['ECAL_Dn']/sf_L1PreFiring
    events['weightcorr_L1PreFiring_Muon_Central_rel'] = events['L1PreFiringWeight']['Muon_Nom']/sf_L1PreFiring
    events['weightcorr_L1PreFiring_Muon_StatUp_rel'] = events['L1PreFiringWeight']['Muon_StatUp']/sf_L1PreFiring
    events['weightcorr_L1PreFiring_Muon_StatDown_rel'] = events['L1PreFiringWeight']['Muon_StatDn']/sf_L1PreFiring
    events['weightcorr_L1PreFiring_Muon_SystUp_rel'] = events['L1PreFiringWeight']['Muon_SystUp']/sf_L1PreFiring
    events['weightcorr_L1PreFiring_Muon_SystDown_rel'] = events['L1PreFiringWeight']['Muon_SystDn']/sf_L1PreFiring
    return sf_L1PreFiring

def myJetSF(jets, ceval, syst):
    j, nj = ak.flatten(jets), ak.num(jets)
    sf = ceval.evaluate(syst,'L', np.array(j.hadronFlavour), np.array(abs(j.eta)), np.array(j.pt))
    return ak.unflatten(sf, nj)

def get_BTag_sf(events, period):
    # btag wp
    cset = get_correction_central('btag', period)
    #loose WP value
    deepJet_wp_value = cset["deepJet_wp_values"].evaluate("L")

    mask_JetPassLoose = events.bjets.btagDeepFlavB > deepJet_wp_value

    key_effData = {
        0:'deepJet_incl',
        4:'deepCSV_comb', # or deepCSV_mujets?
        5:'deepCSV_comb' # or deepCSV_mujets?
    }
    effMC = {}
    effData = {}
    mask_JetFlavor = {}
    flavors = [0,4,5]
    systematics = ["central", "up_uncorrelated", "up_correlated", "down_uncorrelated", "down_correlated"]
    for flavor in flavors:
        mask_JetFlavor[flavor] = events.bjets.hadronFlavour == flavor
        f_path = os.path.join(os.getenv("ANALYSIS_PATH"),f'CoffeaAnalysis/corrections/BTagSF/data/btagEff_json/{period}/btagEff_{str(flavor)}.json')
        ceval = correctionlib.CorrectionSet.from_file(f_path)
        # need to do that because jagged array
        wrap_ceval = correctionlib_wrapper(ceval['BTagEff'])
        effMC[flavor] = wrap_ceval(events.bjets.pt, abs(events.bjets.eta))
        effData[flavor] = {}
        for systematic in systematics:
            effData[flavor][systematic] = {}
            effData[flavor][systematic]['PassLoose'] = myJetSF(events.bjets[mask_JetFlavor[flavor] & mask_JetPassLoose], cset[key_effData[flavor]], systematic)
            effData[flavor][systematic]['NotPassLoose'] = myJetSF(events.bjets[mask_JetFlavor[flavor] & ~mask_JetPassLoose], cset[key_effData[flavor]], systematic)

    effMC_JetPassLoose = ak.concatenate([effMC[0][mask_JetFlavor[0] & mask_JetPassLoose], effMC[4][mask_JetFlavor[4] & mask_JetPassLoose], effMC[5][mask_JetFlavor[5] & mask_JetPassLoose ]], axis=1)
    effMC_JetNotPassLoose = ak.concatenate([effMC[0][mask_JetFlavor[0] & ~mask_JetPassLoose], effMC[4][mask_JetFlavor[4] & ~mask_JetPassLoose], effMC[5][mask_JetFlavor[5] & ~mask_JetPassLoose ]], axis=1)

    effMC_JetPassLoose_btagSFbc = ak.concatenate([effMC[4][mask_JetFlavor[4] & mask_JetPassLoose], effMC[5][mask_JetFlavor[5] & mask_JetPassLoose ]], axis=1)
    effMC_JetNotPassLoose_btagSFbc = ak.concatenate([effMC[4][mask_JetFlavor[4] & ~mask_JetPassLoose], effMC[5][mask_JetFlavor[5] & ~mask_JetPassLoose ]], axis=1)

    effMC_JetPassLoose_btagSFligh = effMC[0][mask_JetFlavor[0] & mask_JetPassLoose]
    effMC_JetNotPassLoose_btagSFligh = effMC[0][mask_JetFlavor[0] & ~mask_JetPassLoose]

    P_MC = ak.prod(effMC_JetPassLoose, axis=-1)*ak.prod(1-effMC_JetNotPassLoose, axis=-1)
    P_MC_btagSFbc = ak.prod(effMC_JetPassLoose_btagSFbc, axis=-1)*ak.prod(1-effMC_JetNotPassLoose_btagSFbc, axis=-1)
    P_MC_btagSFligh = ak.prod(effMC_JetPassLoose_btagSFligh, axis=-1)*ak.prod(1-effMC_JetNotPassLoose_btagSFligh, axis=-1)

    BTag_sf = {}
    BTag_sf_btagSFbc = {}
    BTag_sf_btagSFligh = {}
    for systematic in systematics:
        effData_JetPassLoose = ak.concatenate([effData[0][systematic]['PassLoose'], effData[4][systematic]['PassLoose'], effData[5][systematic]['PassLoose']], axis=-1)
        effData_JetNotPassLoose = ak.concatenate([effData[0][systematic]['NotPassLoose'], effData[4][systematic]['NotPassLoose'], effData[5][systematic]['NotPassLoose']], axis=-1)

        effData_JetPassLoose_btagSFbc = ak.concatenate([effData[4][systematic]['PassLoose'], effData[5][systematic]['PassLoose']], axis=-1)
        effData_JetNotPassLoose_btagSFbc = ak.concatenate([effData[4][systematic]['NotPassLoose'], effData[5][systematic]['NotPassLoose']], axis=-1)

        effData_JetPassLoose_btagSFligh = effData[0][systematic]['PassLoose']
        effData_JetNotPassLoose_btagSFligh = effData[0][systematic]['NotPassLoose']

        P_data = ak.prod(effMC_JetPassLoose*effData_JetPassLoose, axis=-1)*ak.prod(1-(effMC_JetNotPassLoose*effData_JetNotPassLoose), axis=-1)
        P_data_btagSFbc = ak.prod(effMC_JetPassLoose_btagSFbc*effData_JetPassLoose_btagSFbc, axis=-1)*ak.prod(1-(effMC_JetNotPassLoose_btagSFbc*effData_JetNotPassLoose_btagSFbc), axis=-1)
        P_data_btagSFligh = ak.prod(effMC_JetPassLoose_btagSFligh*effData_JetPassLoose_btagSFligh, axis=-1)*ak.prod(1-(effMC_JetNotPassLoose_btagSFligh*effData_JetNotPassLoose_btagSFligh), axis=-1)
        
        BTag_sf[systematic] = P_data/P_MC
        BTag_sf_btagSFbc[systematic] = P_data_btagSFbc/P_MC_btagSFbc
        BTag_sf_btagSFligh[systematic] = P_data_btagSFligh/P_MC_btagSFligh

    events[f'weightcorr_bTagSF_Loose_Total_Central'] = BTag_sf["central"]
    events[f'weightcorr_bTagSF_Loose_btagSFbc_corr_Up_rel'] = BTag_sf_btagSFbc["up_correlated"]/BTag_sf["central"]
    events[f'weightcorr_bTagSF_Loose_btagSFbc_corr_Down_rel'] = BTag_sf_btagSFbc["down_correlated"]/BTag_sf["central"]
    events[f'weightcorr_bTagSF_Loose_btagSFbc_{period}_Up_rel'] = BTag_sf_btagSFbc["up_uncorrelated"]/BTag_sf["central"]
    events[f'weightcorr_bTagSF_Loose_btagSFbc_{period}_Down_rel'] = BTag_sf_btagSFbc["down_uncorrelated"]/BTag_sf["central"]
    events[f'weightcorr_bTagSF_Loose_btagSFlight_corr_Up_rel'] = BTag_sf_btagSFligh["up_correlated"]/BTag_sf["central"]
    events[f'weightcorr_bTagSF_Loose_btagSFlight_corr_Down_rel'] = BTag_sf_btagSFligh["down_correlated"]/BTag_sf["central"]
    events[f'weightcorr_bTagSF_Loose_btagSFlight_{period}_Up_rel'] = BTag_sf_btagSFligh["up_uncorrelated"]/BTag_sf["central"]
    events[f'weightcorr_bTagSF_Loose_btagSFlight_{period}_Down_rel'] = BTag_sf_btagSFligh["down_uncorrelated"]/BTag_sf["central"]

    return BTag_sf["central"]

def compute_jet_corr(events, period, mode):
    if mode != 'Data':
        path_to_corr = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/JET/data/MC'
        Area_JEC = {
            '2016_HIPM': 'Summer19UL16APV_V7_MC',
            '2016': 'Summer19UL16_V7_MC',
            '2017': 'Summer19UL17_V5_MC',
            '2018': 'Summer19UL18_V5_MC'
        }
        Area_JER = {
            '2016_HIPM': 'Summer20UL16APV_JRV3_MC',
            '2016': 'Summer20UL16_JRV3_MC',
            '2017': 'Summer19UL17_JRV2_MC',
            '2018': 'Summer19UL18_JRV2_MC'
        }
    else:
        path_to_corr = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/JET/data/DATA'
        ds = events.metadata["dataset"] # dataset name
        ds = ds.split('_nano')[0]
        AREA = ds[-1:]

        if period in ['2017', '2018']:
            Area_JEC = {
                '2017': f'Summer19UL17_Run{AREA}_V5_DATA',
                '2018': f'Summer19UL18_Run{AREA}_V5_DATA'}
        if period == '2016':
            Area_JEC = {'2016': 'Summer19UL16_RunFGH_V7_DATA'}
        if period == '2016_HIPM':
            if AREA in ['B','C','D']:
                Area_JEC = {'2016_HIPM': 'Summer19UL16APV_RunBCD_V7_DATA'}
            if AREA in ['E','F']:
                Area_JEC = {'2016_HIPM': 'Summer19UL16APV_RunEF_V7_DATA'}
        Area_JER = {
            '2016_HIPM': 'Summer20UL16APV_JRV3_DATA',
            '2016': 'Summer20UL16_JRV3_DATA',
            '2017': 'Summer19UL17_JRV2_DATA',
            '2018': 'Summer19UL18_JRV2_DATA'
        }
        
    type_of_jet = 'AK4PFchs'
    corrections_names_JEC = ['L1FastJet', 'L2Relative', 'L3Absolute', 'L2L3Residual']
    corrections_names_JER = ['PtResolution', 'SF']

    ext = extractor()
    ext.add_weight_sets([
        f" * * {path_to_corr}/{Area_JEC[period]}/{Area_JEC[period]}_{corr_name}_{type_of_jet}.txt" for corr_name in corrections_names_JEC
    ])
    ext.add_weight_sets([f" * * {path_to_corr}/{Area_JEC[period]}/{Area_JEC[period]}_Uncertainty_{type_of_jet}.junc.txt"])
    ext.add_weight_sets([
        f" * * {path_to_corr}/{Area_JER[period]}/{Area_JER[period]}_{corr_name}_{type_of_jet}.txt" for corr_name in corrections_names_JER
    ])
    ext.finalize()
    evaluator = ext.make_evaluator()

    inputs = {name: evaluator[name] for name in dir(evaluator)}
    stack = JECStack(inputs)
    
    #jetVetoMap = GetJetVetoMaps(events, period)
    #events["SelJet","vetomap"] = jetVetoMap

    jets = events.SelJet
    jets['pt_raw'] = (1 - jets['rawFactor']) * jets['pt']
    jets['mass_raw'] = (1 - jets['rawFactor']) * jets['mass']
    jets['PU_rho'] = ak.broadcast_arrays(events.Rho.fixedGridRhoFastjetAll, jets.pt)[0]
    #jets['vetoMap'] = jetVetoMap
    name_map = stack.blank_name_map
    name_map['JetPt'] = 'pt'
    name_map['JetEta'] = 'eta'
    name_map['JetPhi'] = 'phi'
    name_map['JetMass'] = 'mass'
    name_map['JetA'] = 'area'
    name_map['ptRaw'] = 'pt_raw'
    name_map['massRaw'] = 'mass_raw'
    name_map['Rho'] = 'PU_rho'
    #name_map['vetoMap'] = 'vetoMap'
    name_map['METpt'] = 'pt'
    name_map['METphi'] = 'phi'
    name_map['UnClusteredEnergyDeltaX'] = 'MetUnclustEnUpDeltaX'
    name_map['UnClusteredEnergyDeltaY'] = 'MetUnclustEnUpDeltaY'
    
    events_cache = events.caches[0]
    if mode != 'Data':
        jets['pt_gen'] = ak.values_astype(ak.fill_none(jets.MatchedGenPt, 0), np.float32)
        name_map['ptGenJet'] = 'pt_gen'
    else:
        #JER sf for data are set to one but we need moke ptGenJet to apply CorrectedJetsFactory to DATA data
        jets['pt_gen'] = ak.values_astype(ak.fill_none(jets.pt*0, 0), np.float32)
        name_map['ptGenJet'] = 'pt_gen'

    jet_factory = CorrectedJetsFactory(name_map, stack)
    if len(jets) == 0:
        return jets
    else:
        corrected_jets = jet_factory.build(jets, lazy_cache=events_cache)
        return corrected_jets 


#helpers ----------------------------------------------------------------------------------------------------------------------
def weightcorr_TauID_genuineElectron(events, Sel_Tau, name, ceval, period, DeepTauVersion):
    sf_Vse = ceval[f"{DeepTauVersion}VSe"].evaluate(Sel_Tau.eta, Sel_Tau.genPartFlav, 'VVLoose', 'nom')
    sf_Vse_up_rel = ceval[f"{DeepTauVersion}VSe"].evaluate(Sel_Tau.eta, Sel_Tau.genPartFlav, 'VVLoose', 'up')/sf_Vse
    sf_Vse_down_rel = ceval[f"{DeepTauVersion}VSe"].evaluate(Sel_Tau.eta, Sel_Tau.genPartFlav, 'VVLoose', 'down')/sf_Vse
    #SFs are eta dependent (split into barrel and endcap regions, i.e [0,1.5,2.3]), and uncorelated across area
    Sel_Tau_barrel = (np.abs(Sel_Tau.eta) < 1.5)
    Sel_Tau_endcaps = (np.abs(Sel_Tau.eta) >= 1.5)
    #save in events
    events[f'weightcorr_{name}_TauID_genuineElectron_{period}_barrel_Up_rel']    = np.where(Sel_Tau_barrel, sf_Vse_up_rel,   np.ones_like(sf_Vse))
    events[f'weightcorr_{name}_TauID_genuineElectron_{period}_barrel_Down_rel']  = np.where(Sel_Tau_barrel, sf_Vse_down_rel, np.ones_like(sf_Vse))
    events[f'weightcorr_{name}_TauID_genuineElectron_{period}_endcaps_Up_rel']   = np.where(Sel_Tau_endcaps, sf_Vse_up_rel,  np.ones_like(sf_Vse))
    events[f'weightcorr_{name}_TauID_genuineElectron_{period}_endcaps_Down_rel'] = np.where(Sel_Tau_endcaps, sf_Vse_down_rel, np.ones_like(sf_Vse))
    events[f'weightcorr_{name}_TauID_genuineElectron_Total_Central'] = sf_Vse
    return sf_Vse

def weightcorr_TauID_genuineMuon(events, Sel_Tau, name, ceval, period, DeepTauVersion):
    sf_Vsmu = ceval[f"{DeepTauVersion}VSmu"].evaluate(Sel_Tau.eta, Sel_Tau.genPartFlav, 'Tight', 'nom')
    sf_Vsmu_up_rel = ceval[f"{DeepTauVersion}VSmu"].evaluate(Sel_Tau.eta, Sel_Tau.genPartFlav, 'Tight', 'up')/sf_Vsmu
    sf_Vsmu_down_rel = ceval[f"{DeepTauVersion}VSmu"].evaluate(Sel_Tau.eta, Sel_Tau.genPartFlav, 'Tight', 'down')/sf_Vsmu
    #SFs are eta dependent (bins are [0,0.4,0.8,1.2,1.7,2.3]), and uncorelated across area 
    Sel_Tau_etaLt0p4 = (np.abs(Sel_Tau.eta) < 0.4)
    Sel_Tau_eta0p4to0p8 = ((np.abs(Sel_Tau.eta) >= 0.4) & (np.abs(Sel_Tau.eta) < 0.8))
    Sel_Tau_eta0p8to1p2 = ((np.abs(Sel_Tau.eta) >= 0.8) & (np.abs(Sel_Tau.eta) < 1.2))
    Sel_Tau_eta1p2to1p7 = ((np.abs(Sel_Tau.eta) >= 1.2) & (np.abs(Sel_Tau.eta) < 1.7))
    Sel_Tau_etaGt1p7 = (np.abs(Sel_Tau.eta) >= 1.7)
    #save in events
    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_etaLt0p4_Up_rel']     = np.where(Sel_Tau_etaLt0p4   , sf_Vsmu_up_rel,   np.ones_like(sf_Vsmu))
    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_etaLt0p4_Down_rel']   = np.where(Sel_Tau_etaLt0p4   , sf_Vsmu_down_rel, np.ones_like(sf_Vsmu))
    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_eta0p4to0p8_Up_rel']  = np.where(Sel_Tau_eta0p4to0p8, sf_Vsmu_up_rel,   np.ones_like(sf_Vsmu))
    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_eta0p4to0p8_Down_rel']= np.where(Sel_Tau_eta0p4to0p8, sf_Vsmu_down_rel, np.ones_like(sf_Vsmu))
    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_eta0p8to1p2_Up_rel']  = np.where(Sel_Tau_eta0p8to1p2, sf_Vsmu_up_rel,   np.ones_like(sf_Vsmu))
    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_eta0p8to1p2_Down_rel']= np.where(Sel_Tau_eta0p8to1p2, sf_Vsmu_down_rel, np.ones_like(sf_Vsmu))
    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_eta1p2to1p7_Up_rel']  = np.where(Sel_Tau_eta1p2to1p7, sf_Vsmu_up_rel,   np.ones_like(sf_Vsmu))
    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_eta1p2to1p7_Down_rel']= np.where(Sel_Tau_eta1p2to1p7, sf_Vsmu_down_rel, np.ones_like(sf_Vsmu))
    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_etaGt1p7_Up_rel']     = np.where(Sel_Tau_etaGt1p7   , sf_Vsmu_up_rel,   np.ones_like(sf_Vsmu))
    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_etaGt1p7_Down_rel']   = np.where(Sel_Tau_etaGt1p7   , sf_Vsmu_down_rel, np.ones_like(sf_Vsmu))
    events[f'weightcorr_{name}_TauID_genuineMuon_Total_Central'] = sf_Vsmu
    return sf_Vsmu

def weightcorr_TauID_genuineTau(events, Sel_Tau, name, ceval, period, DeepTauVersion):
    sf_VsJet = ceval[f"{DeepTauVersion}VSjet"].evaluate(Sel_Tau.pt, Sel_Tau.decayMode, Sel_Tau.genPartFlav, 'Medium', 'VVLoose', 'nom', 'dm')
    # SFs from the uncertainties on the linear fit parameters
    #   - Uncstat1, Uncstat2 = statistical uncertainties (Uncstat{i}Fit) decorrelated by DM (binned in 0, 1, 10, and 11) and era
    DecayModes = ['0', '1', '10', '11']
    stat_sf_list = ['stat1', 'stat2']
    for stat in stat_sf_list:
        for dm in DecayModes:
            sf_VsJet_Up_rel = ceval[f"{DeepTauVersion}VSjet"].evaluate(Sel_Tau.pt, Sel_Tau.decayMode, Sel_Tau.genPartFlav, 'Medium', 'VVLoose', f'{stat}_dm{dm}_up', 'dm')/sf_VsJet
            sf_VsJet_Down_rel = ceval[f"{DeepTauVersion}VSjet"].evaluate(Sel_Tau.pt, Sel_Tau.decayMode, Sel_Tau.genPartFlav, 'Medium', 'VVLoose', f'{stat}_dm{dm}_down', 'dm')/sf_VsJet
            Sel_Tau_dm = (Sel_Tau.decayMode == int(dm))
            events[f'weightcorr_{name}_TauID_genuineTau_Unc{stat}_DM{dm}_{period}_Up_rel'] =   np.where(Sel_Tau_dm   , sf_VsJet_Up_rel,   np.ones_like(sf_VsJet))
            events[f'weightcorr_{name}_TauID_genuineTau_Unc{stat}_DM{dm}_{period}_Down_rel'] = np.where(Sel_Tau_dm   , sf_VsJet_Down_rel, np.ones_like(sf_VsJet))
    #   - UncSystAllEras = The component of the systematic uncertainty that is correlated across DMs and eras
    events[f'weightcorr_{name}_TauID_genuineTau_UncSystAllEras_Up_rel'] = ceval[f"{DeepTauVersion}VSjet"].evaluate(Sel_Tau.pt, Sel_Tau.decayMode, Sel_Tau.genPartFlav, 'Medium', 'VVLoose', 'syst_alleras_up', 'dm')/sf_VsJet 
    events[f'weightcorr_{name}_TauID_genuineTau_UncSystAllEras_Down_rel'] = ceval[f"{DeepTauVersion}VSjet"].evaluate(Sel_Tau.pt, Sel_Tau.decayMode, Sel_Tau.genPartFlav, 'Medium', 'VVLoose', 'syst_alleras_down', 'dm')/sf_VsJet 
    #   - UncSyst{Era} = The component of the systematic uncertainty that is correlated across DMs but uncorrelated by eras
    events[f'weightcorr_{name}_TauID_genuineTau_UncSyst_{period}_Up_rel'] = ceval[f"{DeepTauVersion}VSjet"].evaluate(Sel_Tau.pt, Sel_Tau.decayMode, Sel_Tau.genPartFlav, 'Medium', 'VVLoose', f'syst_{Area_ref[period]}_up', 'dm')/sf_VsJet 
    events[f'weightcorr_{name}_TauID_genuineTau_UncSyst_{period}_Down_rel'] = ceval[f"{DeepTauVersion}VSjet"].evaluate(Sel_Tau.pt, Sel_Tau.decayMode, Sel_Tau.genPartFlav, 'Medium', 'VVLoose', f'syst_{Area_ref[period]}_down', 'dm')/sf_VsJet 
    #   - UncSyst{Era}_DM{DM} = The component of the systematic uncertainty due to the tau energy scale that is correlated across DMs and eras
    for dm in DecayModes:
        sf_VsJet_Up_rel = ceval[f"{DeepTauVersion}VSjet"].evaluate(Sel_Tau.pt, Sel_Tau.decayMode, Sel_Tau.genPartFlav, 'Medium', 'VVLoose', f'syst_TES_{period}_dm{dm}_up', 'dm')/sf_VsJet
        sf_VsJet_Down_rel = ceval[f"{DeepTauVersion}VSjet"].evaluate(Sel_Tau.pt, Sel_Tau.decayMode, Sel_Tau.genPartFlav, 'Medium', 'VVLoose', f'syst_TES_{period}_dm{dm}_down', 'dm')/sf_VsJet
        Sel_Tau_dm = (Sel_Tau.decayMode == int(dm))
        events[f'weightcorr_{name}_TauID_genuineTau_UncSyst_DM{dm}_{period}_Up_rel'] =  np.where(Sel_Tau_dm   , sf_VsJet_Up_rel,   np.ones_like(sf_VsJet))
        events[f'weightcorr_{name}_TauID_genuineTau_UncSyst_DM{dm}_{period}_Down_rel'] =np.where(Sel_Tau_dm   , sf_VsJet_Down_rel, np.ones_like(sf_VsJet))
    events[f'weightcorr_{name}_TauID_genuineTau_Total_Central'] = sf_VsJet 
    return sf_VsJet

def weightcorr_MuID_RECO(events, Sel_Muon, name, ceval, period):
    # RECO sf are split between muon with pt<120 GeV and above 120
    nlepBelow120 = (Sel_Muon.pt <= 120)
    nlepAbove120 = (Sel_Muon.pt > 120)

    # for Mons with pt<120 GeV, file binning is [40, inf] but the recommendation is to apply them for muons with pT in the range [10; 200] GeV.
    Muon_pt = (Sel_Muon.pt)
    Muon_pt = np.where(Muon_pt<40, 40, Muon_pt)
    RECOsf_Below120 = (ceval['NUM_TrackerMuons_DEN_genTracks'].evaluate(np.abs(Sel_Muon.eta), Muon_pt, "nominal"))
    RECOsf_Below120_syst_Up_rel =   (RECOsf_Below120 + (ceval['NUM_TrackerMuons_DEN_genTracks'].evaluate(np.abs(Sel_Muon.eta), Muon_pt, "syst")))/RECOsf_Below120
    RECOsf_Below120_syst_Down_rel = (RECOsf_Below120 - (ceval['NUM_TrackerMuons_DEN_genTracks'].evaluate(np.abs(Sel_Muon.eta), Muon_pt, "syst")))/RECOsf_Below120
    RECOsf_Below120_stat_Up_rel =   (RECOsf_Below120 + (ceval['NUM_TrackerMuons_DEN_genTracks'].evaluate(np.abs(Sel_Muon.eta), Muon_pt, "stat")))/RECOsf_Below120
    RECOsf_Below120_stat_Down_rel = (RECOsf_Below120 - (ceval['NUM_TrackerMuons_DEN_genTracks'].evaluate(np.abs(Sel_Muon.eta), Muon_pt, "stat")))/RECOsf_Below120

    # for Mons with pt>120 GeV, custom tab from twiki because sf are not in muon_Z.json
    Muon_p = (Sel_Muon.p)
    Muon_p = np.where(Muon_p<50, 50, Muon_p)
    RECOevaluator_Above120 = correctionlib.CorrectionSet.from_file(f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/muon/data/'+period+f'/ScaleFactors_Muon_highPt_RECO_{Area_ref[period]}_schemaV2.json')
    RECOsf_Above120 = (RECOevaluator_Above120["NUM_GlobalMuons_DEN_TrackerMuonProbes"].evaluate(np.abs(Sel_Muon.eta), Muon_p, 'nominal'))
    RECOsf_Above120_syst_Up_rel =   (RECOsf_Above120 + (RECOevaluator_Above120["NUM_GlobalMuons_DEN_TrackerMuonProbes"].evaluate(np.abs(Sel_Muon.eta), Muon_p, 'syst')))/RECOsf_Above120
    RECOsf_Above120_syst_Down_rel = (RECOsf_Above120 - (RECOevaluator_Above120["NUM_GlobalMuons_DEN_TrackerMuonProbes"].evaluate(np.abs(Sel_Muon.eta), Muon_p, 'syst')))/RECOsf_Above120
    RECOsf_Above120_stat_Up_rel =   (RECOsf_Above120 + (RECOevaluator_Above120["NUM_GlobalMuons_DEN_TrackerMuonProbes"].evaluate(np.abs(Sel_Muon.eta), Muon_p, 'stat')))/RECOsf_Above120
    RECOsf_Above120_stat_Down_rel = (RECOsf_Above120 - (RECOevaluator_Above120["NUM_GlobalMuons_DEN_TrackerMuonProbes"].evaluate(np.abs(Sel_Muon.eta), Muon_p, 'stat')))/RECOsf_Above120

    RECO_sf = np.where(nlepBelow120, RECOsf_Below120,   RECOsf_Above120)

    #save in events
    events[f'weightcorr_{name}_MuID_RECO_ptlt120_syst_Up_rel']    = np.where(nlepBelow120, RECOsf_Below120_syst_Up_rel,   np.ones_like(RECO_sf))
    events[f'weightcorr_{name}_MuID_RECO_ptlt120_syst_Down_rel']  = np.where(nlepBelow120, RECOsf_Below120_syst_Down_rel, np.ones_like(RECO_sf))
    events[f'weightcorr_{name}_MuID_RECO_ptlt120_stat_{period}_Up_rel']   = np.where(nlepBelow120, RECOsf_Below120_stat_Up_rel,  np.ones_like(RECO_sf))
    events[f'weightcorr_{name}_MuID_RECO_ptlt120_stat_{period}_Down_rel'] = np.where(nlepBelow120, RECOsf_Below120_stat_Down_rel, np.ones_like(RECO_sf))

    events[f'weightcorr_{name}_MuID_RECO_ptgt120_syst_Up_rel'] = np.where(nlepAbove120, RECOsf_Above120_syst_Up_rel,   np.ones_like(RECO_sf))
    events[f'weightcorr_{name}_MuID_RECO_ptgt120_syst_Down_rel'] = np.where(nlepAbove120, RECOsf_Above120_syst_Down_rel,   np.ones_like(RECO_sf))
    events[f'weightcorr_{name}_MuID_RECO_ptgt120_stat_{period}_Up_rel'] = np.where(nlepAbove120, RECOsf_Above120_stat_Up_rel,   np.ones_like(RECO_sf))
    events[f'weightcorr_{name}_MuID_RECO_ptgt120_stat_{period}_Down_rel'] = np.where(nlepAbove120, RECOsf_Above120_stat_Down_rel,   np.ones_like(RECO_sf))

    return RECO_sf

def weightcorr_MuID_MediumID(events, Sel_Muon, name, ceval, period):
    # ID sf are split between muon with pt<120 GeV and above 120
    nlepBelow120 = (Sel_Muon.pt <= 120)
    nlepAbove120 = (Sel_Muon.pt > 120)

    IDsf_Below120 = (ceval['NUM_MediumID_DEN_TrackerMuons'].evaluate(np.abs(Sel_Muon.eta), Sel_Muon.pt, 'nominal'))
    IDsf_Below120_syst_Up_rel =   (IDsf_Below120 + (ceval['NUM_MediumID_DEN_TrackerMuons'].evaluate(np.abs(Sel_Muon.eta), Sel_Muon.pt, "syst")))/IDsf_Below120
    IDsf_Below120_syst_Down_rel = (IDsf_Below120 - (ceval['NUM_MediumID_DEN_TrackerMuons'].evaluate(np.abs(Sel_Muon.eta), Sel_Muon.pt, "syst")))/IDsf_Below120
    IDsf_Below120_stat_Up_rel =   (IDsf_Below120 + (ceval['NUM_MediumID_DEN_TrackerMuons'].evaluate(np.abs(Sel_Muon.eta), Sel_Muon.pt, "stat")))/IDsf_Below120
    IDsf_Below120_stat_Down_rel = (IDsf_Below120 - (ceval['NUM_MediumID_DEN_TrackerMuons'].evaluate(np.abs(Sel_Muon.eta), Sel_Muon.pt, "stat")))/IDsf_Below120

    # for Mons with pt>120 GeV, custom tab from twiki because sf are not in muon_Z.json.
    Muon_pt = (Sel_Muon.pt)
    Muon_pt = np.where(Muon_pt<50, 50, Muon_pt)
    IDevaluator_Above120 = correctionlib.CorrectionSet.from_file(f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/muon/data/'+period+f'/ScaleFactors_Muon_highPt_IDISO_{Area_ref[period]}_schemaV2.json')
    IDsf_Above120 = (IDevaluator_Above120["NUM_MediumID_DEN_GlobalMuonProbes"].evaluate(np.abs(Sel_Muon.eta), Muon_pt, 'nominal'))
    IDsf_Above120_syst_Up_rel =   (IDsf_Above120 + (IDevaluator_Above120["NUM_MediumID_DEN_GlobalMuonProbes"].evaluate(np.abs(Sel_Muon.eta), Muon_pt, 'syst')))/IDsf_Above120
    IDsf_Above120_syst_Down_rel = (IDsf_Above120 - (IDevaluator_Above120["NUM_MediumID_DEN_GlobalMuonProbes"].evaluate(np.abs(Sel_Muon.eta), Muon_pt, 'syst')))/IDsf_Above120
    IDsf_Above120_stat_Up_rel =   (IDsf_Above120 + (IDevaluator_Above120["NUM_MediumID_DEN_GlobalMuonProbes"].evaluate(np.abs(Sel_Muon.eta), Muon_pt, 'stat')))/IDsf_Above120
    IDsf_Above120_stat_Down_rel = (IDsf_Above120 - (IDevaluator_Above120["NUM_MediumID_DEN_GlobalMuonProbes"].evaluate(np.abs(Sel_Muon.eta), Muon_pt, 'stat')))/IDsf_Above120

    ID_sf = np.where(nlepBelow120, IDsf_Below120,   IDsf_Above120)

    #save in events
    events[f'weightcorr_{name}_MuID_MediumID_ptlt120_syst_Up_rel']    = np.where(nlepBelow120, IDsf_Below120_syst_Up_rel,   np.ones_like(ID_sf))
    events[f'weightcorr_{name}_MuID_MediumID_ptlt120_syst_Down_rel']  = np.where(nlepBelow120, IDsf_Below120_syst_Down_rel, np.ones_like(ID_sf))
    events[f'weightcorr_{name}_MuID_MediumID_ptlt120_stat_{period}_Up_rel']   = np.where(nlepBelow120, IDsf_Below120_stat_Up_rel,  np.ones_like(ID_sf))
    events[f'weightcorr_{name}_MuID_MediumID_ptlt120_stat_{period}_Down_rel'] = np.where(nlepBelow120, IDsf_Below120_stat_Down_rel, np.ones_like(ID_sf))

    events[f'weightcorr_{name}_MuID_MediumID_ptgt120_syst_Up_rel'] = np.where(nlepAbove120, IDsf_Above120_syst_Up_rel,   np.ones_like(ID_sf))
    events[f'weightcorr_{name}_MuID_MediumID_ptgt120_syst_Down_rel'] = np.where(nlepAbove120, IDsf_Above120_syst_Down_rel,   np.ones_like(ID_sf))
    events[f'weightcorr_{name}_MuID_MediumID_ptgt120_stat_{period}_Up_rel'] = np.where(nlepAbove120, IDsf_Above120_stat_Up_rel,   np.ones_like(ID_sf))
    events[f'weightcorr_{name}_MuID_MediumID_ptgt120_stat_{period}_Down_rel'] = np.where(nlepAbove120, IDsf_Above120_stat_Down_rel,   np.ones_like(ID_sf))

    return ID_sf

def weightcorr_MuID_LooseISO(events, Sel_Muon, name, ceval, period):
    # ISO sf are split between muon with pt<120 GeV and above 120
    nlepBelow120 = (Sel_Muon.pt <= 120)
    nlepAbove120 = (Sel_Muon.pt > 120)

    ISOsf_Below120 = (ceval['NUM_LooseRelIso_DEN_MediumID'].evaluate(np.abs(Sel_Muon.eta), Sel_Muon.pt, 'nominal'))
    ISOsf_Below120_syst_Up_rel =   (ISOsf_Below120 + (ceval['NUM_LooseRelIso_DEN_MediumID'].evaluate(np.abs(Sel_Muon.eta), Sel_Muon.pt, "syst")))/ISOsf_Below120
    ISOsf_Below120_syst_Down_rel = (ISOsf_Below120 - (ceval['NUM_LooseRelIso_DEN_MediumID'].evaluate(np.abs(Sel_Muon.eta), Sel_Muon.pt, "syst")))/ISOsf_Below120
    ISOsf_Below120_stat_Up_rel =   (ISOsf_Below120 + (ceval['NUM_LooseRelIso_DEN_MediumID'].evaluate(np.abs(Sel_Muon.eta), Sel_Muon.pt, "stat")))/ISOsf_Below120
    ISOsf_Below120_stat_Down_rel = (ISOsf_Below120 - (ceval['NUM_LooseRelIso_DEN_MediumID'].evaluate(np.abs(Sel_Muon.eta), Sel_Muon.pt, "stat")))/ISOsf_Below120

    # for Mons with pt>120 GeV, custom tab from twiki because sf are not in muon_Z.json. 
    Muon_pt = (Sel_Muon.pt)
    Muon_pt = np.where(Muon_pt<50, 50, Muon_pt)
    ISOevaluator_Above120 = correctionlib.CorrectionSet.from_file(f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/muon/data/'+period+f'/ScaleFactors_Muon_highPt_IDISO_{Area_ref[period]}_schemaV2.json')
    ISOsf_Above120 = (ISOevaluator_Above120["NUM_probe_LooseRelTkIso_DEN_MediumIDProbes"].evaluate(np.abs(Sel_Muon.eta), Muon_pt, 'nominal'))
    ISOsf_Above120_syst_Up_rel =   (ISOsf_Above120 + (ISOevaluator_Above120["NUM_probe_LooseRelTkIso_DEN_MediumIDProbes"].evaluate(np.abs(Sel_Muon.eta), Muon_pt, 'syst')))/ISOsf_Above120
    ISOsf_Above120_syst_Down_rel = (ISOsf_Above120 - (ISOevaluator_Above120["NUM_probe_LooseRelTkIso_DEN_MediumIDProbes"].evaluate(np.abs(Sel_Muon.eta), Muon_pt, 'syst')))/ISOsf_Above120
    ISOsf_Above120_stat_Up_rel =   (ISOsf_Above120 + (ISOevaluator_Above120["NUM_probe_LooseRelTkIso_DEN_MediumIDProbes"].evaluate(np.abs(Sel_Muon.eta), Muon_pt, 'stat')))/ISOsf_Above120
    ISOsf_Above120_stat_Down_rel = (ISOsf_Above120 - (ISOevaluator_Above120["NUM_probe_LooseRelTkIso_DEN_MediumIDProbes"].evaluate(np.abs(Sel_Muon.eta), Muon_pt, 'stat')))/ISOsf_Above120

    ISO_sf = np.where(nlepBelow120, ISOsf_Below120,   ISOsf_Above120)

    #save in events
    events[f'weightcorr_{name}_MuID_MediumID_ptlt120_syst_Up_rel']    = np.where(nlepBelow120, ISOsf_Below120_syst_Up_rel,   np.ones_like(ISO_sf))
    events[f'weightcorr_{name}_MuID_MediumID_ptlt120_syst_Down_rel']  = np.where(nlepBelow120, ISOsf_Below120_syst_Down_rel, np.ones_like(ISO_sf))
    events[f'weightcorr_{name}_MuID_MediumID_ptlt120_stat_{period}_Up_rel']   = np.where(nlepBelow120, ISOsf_Below120_stat_Up_rel,  np.ones_like(ISO_sf))
    events[f'weightcorr_{name}_MuID_MediumID_ptlt120_stat_{period}_Down_rel'] = np.where(nlepBelow120, ISOsf_Below120_stat_Down_rel, np.ones_like(ISO_sf))

    events[f'weightcorr_{name}_MuID_MediumID_ptgt120_syst_Up_rel'] = np.where(nlepAbove120, ISOsf_Above120_syst_Up_rel,   np.ones_like(ISO_sf))
    events[f'weightcorr_{name}_MuID_MediumID_ptgt120_syst_Down_rel'] = np.where(nlepAbove120, ISOsf_Above120_syst_Down_rel,   np.ones_like(ISO_sf))
    events[f'weightcorr_{name}_MuID_MediumID_ptgt120_stat_{period}_Up_rel'] = np.where(nlepAbove120, ISOsf_Above120_stat_Up_rel,   np.ones_like(ISO_sf))
    events[f'weightcorr_{name}_MuID_MediumID_ptgt120_stat_{period}_Down_rel'] = np.where(nlepAbove120, ISOsf_Above120_stat_Down_rel,   np.ones_like(ISO_sf))

    return ISO_sf

def weightcorr_EleID_RECO(events, Sel_Electron, name, period):
    file_name = {
        '2018': 'UL2018',
        '2017': 'UL2017',
        '2016': 'UL2016postVFP',
        '2016_HIPM': 'UL2016preVFP',
    }
    # RECO sf are split between electron with pt<20 GeV and above 20
    nlepBelow20 = (Sel_Electron.pt <= 20)
    nlepAbove20 = (Sel_Electron.pt > 20)

    # for Electron with pt<20 GeV
    fReco_path_ptBelow20 = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/electron/data/{period}/egammaEffi_ptBelow20-txt_EGM2D_{file_name[period]}_witherr.root' 
    evaluator_Below20 = get_scales_fromjson(fReco_path_ptBelow20)
    RECOsf_Below20 = evaluator_Below20["EGamma_SF2D"]((Sel_Electron.pt), (Sel_Electron.eta))
    RECOsf_Below20_err = evaluator_Below20["EGamma_SF2D_err"]((Sel_Electron.pt), (Sel_Electron.eta))
    RECOsf_Below20_Up = RECOsf_Below20 + RECOsf_Below20_err
    RECOsf_Below20_Down = RECOsf_Below20 - RECOsf_Below20_err

    # for Electron with pt>20 GeV
    fReco_path_ptAbove20 = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/electron/data/{period}/egammaEffi_ptAbove20-txt_EGM2D_{file_name[period]}_witherr.root' 
    evaluator_Above20 = get_scales_fromjson(fReco_path_ptAbove20)
    Electron_pt = (Sel_Electron.pt)
    Electron_pt = np.where(Electron_pt>500, 500, Electron_pt)
    RECOsf_Above20 = evaluator_Above20["EGamma_SF2D"](Electron_pt, (Sel_Electron.eta))
    RECOsf_Above20_err = evaluator_Above20["EGamma_SF2D_err"](Electron_pt, (Sel_Electron.eta))
    RECOsf_Above20_Up = RECOsf_Above20 + RECOsf_Above20_err
    RECOsf_Above20_Down = RECOsf_Above20 - RECOsf_Above20_err

    #combine SFs
    RECO_sf = np.where(nlepBelow20, RECOsf_Below20,   RECOsf_Above20)
    RECO_sf_Up_rel = np.where(nlepBelow20, RECOsf_Below20_Up,   RECOsf_Above20_Up)/RECO_sf
    RECO_sf_Down_rel = np.where(nlepBelow20, RECOsf_Below20_Down,   RECOsf_Above20_Down)/RECO_sf
    events[f'weightcorr_{name}_EleID_RECO_Up_rel'] = RECO_sf_Up_rel
    events[f'weightcorr_{name}_EleID_RECO_Down_rel'] = RECO_sf_Down_rel

    return RECO_sf

def weightcorr_EleID_wp90noiso(events, Sel_Electron, name, ceval, period):
    #ID sf for e
    ceval = get_correction_central('electron', period)
    ID_sf = ceval['UL-Electron-ID-SF'].evaluate(Area_dir[period][:-3], 'sf', 'wp90noiso', Sel_Electron.eta, Sel_Electron.pt)
    events[f'weightcorr_{name}_EleID_wp90noiso_Up_rel'] = ceval['UL-Electron-ID-SF'].evaluate(Area_dir[period][:-3], 'sfup', 'wp90noiso', Sel_Electron.eta, Sel_Electron.pt) / ID_sf
    events[f'weightcorr_{name}_EleID_wp90noiso_Down_rel'] = ceval['UL-Electron-ID-SF'].evaluate(Area_dir[period][:-3], 'sfdown', 'wp90noiso', Sel_Electron.eta, Sel_Electron.pt) / ID_sf
    return ID_sf

def GetJetVetoMaps(events, period):
    path_to_corr = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/JET/data/vetomaps/'
    Area_veto = {
        '2016_HIPM': 'Summer19UL16_V1',
        '2016': 'Summer19UL16_V1',
        '2017': 'Summer19UL17_V1',
        '2018': 'Summer19UL18_V1'
    }
    fname = path_to_corr + Area_veto[period] + '_jetvetomaps.json'

    # Grab the json
    ceval = correctionlib.CorrectionSet.from_file(fname)

    #Loose jets as recommended in https://cms-jerc.web.cern.ch/Recommendations/#run-2_2
    Is_Loose_jets = (events.SelJet.pt > 15.) & (events.SelJet.jetId >= 2) & ak.where(events.SelJet.pt<50, events.SelJet.puId >= 6,  events.SelJet.puId >= 0)

    # Flatten the inputs
    eta_flat = ak.flatten(events.SelJet.eta)
    phi_flat = ak.flatten(events.SelJet.phi)

    #Put mins and maxes on the accepted values
    eta_flat_bound = ak.where(eta_flat>5.19,5.19,ak.where(eta_flat<-5.19,-5.19,eta_flat))
    phi_flat_bound = ak.where(phi_flat>3.14159,3.14159,ak.where(phi_flat<-3.14159,-3.14159,phi_flat))

    #Get pass/fail values for each jet (0 is pass and >0 is fail)
    jet_vetomap_flat = (ceval[Area_veto[period]].evaluate('jetvetomap',eta_flat_bound,phi_flat_bound) == 0)

    #Unflatten the array
    jet_vetomap = (ak.unflatten(jet_vetomap_flat,ak.num(events.SelJet.phi))) | ~Is_Loose_jets

    return jet_vetomap

def MET_correction(MET, old_obj, corr_obj):
    """
    Correct the MET based on changes to an object (can be tau, jet, ...) transverse momentum.
    This function supports collections of objects for each event.

    Parameters:
    - MET (dict): Original MET with 'pt' and 'phi'.
    - old_obj (object): Collection of objects with original 'pt' and 'phi'.
    - corr_obj (object): Collection of objects with corrected 'pt' and 'phi'.

    Returns:
    - corrected_met (dict): Corrected MET with 'pt' and 'phi'.
    """
    # Compute original MET components
    met_x = MET['pt'] * np.cos(MET['phi'])
    met_y = MET['pt'] * np.sin(MET['phi'])
    
    # Compute old and new obj momentum components
    obj_px_old = old_obj.pt * np.cos(old_obj.phi)
    obj_py_old = old_obj.pt * np.sin(old_obj.phi)
    obj_px_new = corr_obj.pt * np.cos(corr_obj.phi)
    obj_py_new = corr_obj.pt * np.sin(corr_obj.phi)
    
    # Compute the change in obj momentum (sum over all objects per event if collection)
    delta_px = ak.sum(obj_px_old - obj_px_new, axis=-1)
    delta_py = ak.sum(obj_py_old - obj_py_new, axis=-1)
    
    # Apply the correction to MET components
    met_x_corrected = met_x + delta_px
    met_y_corrected = met_y + delta_py
    
    # Recompute corrected MET magnitude and phi
    corrected_met = MET
    corrected_met['pt'] = np.sqrt(met_x_corrected**2 + met_y_corrected**2)
    corrected_met['phi'] = np.arctan2(met_y_corrected, met_x_corrected)
    
    return corrected_met
