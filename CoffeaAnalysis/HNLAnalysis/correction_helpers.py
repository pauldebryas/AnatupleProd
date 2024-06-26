import correctionlib
import awkward as ak
from coffea.lookup_tools import extractor
import os
import numpy as np
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper

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
    # sf for genuine tau (https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2)
    if DeepTauVersion == "DeepTau2018v2p5":
        #load correction from custom json
        f_path = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/{period}/tau/tau_DeepTau2018v2p5_UL{Area_ref[period]}.json'
        ceval = correctionlib.CorrectionSet.from_file(f_path)
        #print(list(ceval.keys())) --> ['DeepTau2018v2p5VSe', 'DeepTau2018v2p5VSjet', 'DeepTau2018v2p5VSmu', 'tau_energy_scale', 'tau_trigger']
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
    return tau_sf

def compute_sf_e(Sel_Electron, events, name, period):
    # sf for electrons (https://twiki.cern.ch/twiki/bin/view/CMS/EgammaUL2016To2018)
     #RECO sf for e
    RECOsf_Below20, RECOsf_Below20_Up, RECOsf_Below20_Down = get_electronRECOsf_Below20(events, Sel_Electron, name, period)
    RECOsf_Above20, RECOsf_Above20_Up, RECOsf_Above20_Down = get_electronRECOsf_Above20(events, Sel_Electron, name, period)
    #combine SFs
    RECO_sf = ak.concatenate([ak.unflatten(RECOsf_Above20, (Sel_Electron.pt >= 20)*1), ak.unflatten(RECOsf_Below20, (Sel_Electron.pt < 20)*1)], axis=1)
    RECO_sf_Up = ak.concatenate([ak.unflatten(RECOsf_Above20_Up, (Sel_Electron.pt >= 20)*1), ak.unflatten(RECOsf_Below20_Up, (Sel_Electron.pt < 20)*1)], axis=1)
    RECO_sf_Down = ak.concatenate([ak.unflatten(RECOsf_Above20_Down, (Sel_Electron.pt >= 20)*1), ak.unflatten(RECOsf_Below20_Down, (Sel_Electron.pt < 20)*1)], axis=1)
    events[f'weightcorr_{name}_EleID_RECO_Up_rel'] = ak.flatten(RECO_sf_Up / RECO_sf)
    events[f'weightcorr_{name}_EleID_RECO_Down_rel'] = ak.flatten(RECO_sf_Down / RECO_sf)

    #ID sf for e
    ceval = get_correction_central('electron', period)
    ID_sf = ceval['UL-Electron-ID-SF'].evaluate(Area_dir[period][:-3], 'sf', 'wp90noiso', ak.to_numpy(Sel_Electron.eta), ak.to_numpy(Sel_Electron.pt))
    events[f'weightcorr_{name}_EleID_wp90noiso_Up_rel'] = ceval['UL-Electron-ID-SF'].evaluate(Area_dir[period][:-3], 'sfup', 'wp90noiso', ak.to_numpy(Sel_Electron.eta), ak.to_numpy(Sel_Electron.pt)) / ID_sf
    events[f'weightcorr_{name}_EleID_wp90noiso_Down_rel'] = ceval['UL-Electron-ID-SF'].evaluate(Area_dir[period][:-3], 'sfdown', 'wp90noiso', ak.to_numpy(Sel_Electron.eta), ak.to_numpy(Sel_Electron.pt)) / ID_sf

    events[f'weightcorr_{name}_EleID_Total_Central'] = ID_sf * ak.flatten(RECO_sf)
    return ID_sf * ak.flatten(RECO_sf)

def compute_sf_mu(Sel_Muon, events, name, period):
    # sf for muon (https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2018)
    ceval = get_correction_central('muon', period)

    #sf_Below20, sf_Below20_Up, sf_Below20_Down = get_muonsf_Below20(events, Sel_Muon, name, ceval, period)
    #sf_Between20and120, sf_Between20and120_Up, sf_Between20and120_Down = get_muonsf_Between20and120(events, Sel_Muon, name, ceval, period)
    sf_Between15and120 = get_muonsf_Between15and120(events, Sel_Muon, name, ceval, period)
    #sf_Above120, sf_Above120_Up, sf_Above120_Down = get_muonsf_Above120(events, Sel_Muon, name, ceval, period)
    sf_Above120 = get_muonsf_Above120(events, Sel_Muon, name, ceval, period)

    #combine SFs
    sf_mu = ak.concatenate([ak.unflatten(sf_Between15and120, ((Sel_Muon.pt > 15) & (Sel_Muon.pt <= 120))*1), 
                            ak.unflatten(sf_Above120,        (Sel_Muon.pt > 120)*1)
                            ], axis=1) #, ak.unflatten(sf_Below20, (Sel_Muon.pt <= 20)*1)
    #sf_mu_Up = ak.concatenate([ak.unflatten(sf_Between20and120_Up, ((Sel_Muon.pt > 20) & (Sel_Muon.pt <= 120))*1), ak.unflatten(sf_Above120_Up, (Sel_Muon.pt > 120)*1), ak.unflatten(sf_Below20_Up, (Sel_Muon.pt <= 20)*1)], axis=1)
    #sf_mu_Down = ak.concatenate([ak.unflatten(sf_Between20and120_Down, ((Sel_Muon.pt > 20) & (Sel_Muon.pt <= 120))*1), ak.unflatten(sf_Above120_Down, (Sel_Muon.pt > 120)*1), ak.unflatten(sf_Below20_Down, (Sel_Muon.pt <= 20)*1)], axis=1)
    events[f'weightcorr_{name}_MuID_Total_Central'] = sf_mu
    #events[f'weightcorr_{name}_MuID_Total_Up'] =sf_mu_Up
    #events[f'weightcorr_{name}_MuID_Total_Down'] =sf_mu_Down
    return ak.flatten(sf_mu)

def get_trigger_correction_mu(Sel_Muon, events, name, period):
    HLT_name = {
        '2018': 'NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight',
        '2017': 'NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight',
        '2016_HIPM': 'NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight',
        '2016': 'NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight',
    }
    if period == '2017': #IsoMu27
        ev_pt = ak.to_numpy(Sel_Muon.pt)
        ev_pt = np.where(ev_pt<29, 29, ev_pt)
    else: #IsoMu24
        ev_pt = ak.to_numpy(Sel_Muon.pt)
        ev_pt = np.where(ev_pt<26, 26, ev_pt)
    ceval = get_correction_central('muon', period)
    TrgSF = ceval[HLT_name[period]].evaluate(ak.to_numpy(abs(Sel_Muon.eta)), ak.to_numpy(ev_pt), "nominal")
    events[f'weightcorr_{name}_TrgSF_singleMu_Total_Central'] = TrgSF
    events[f'weightcorr_{name}_TrgSF_singleMu_syst_Up_rel'] =    (TrgSF + ceval[HLT_name[period]].evaluate(ak.to_numpy(abs(Sel_Muon.eta)), ak.to_numpy(ev_pt), "syst"))/TrgSF
    events[f'weightcorr_{name}_TrgSF_singleMu_syst_Down_rel'] =  (TrgSF - ceval[HLT_name[period]].evaluate(ak.to_numpy(abs(Sel_Muon.eta)), ak.to_numpy(ev_pt), "syst"))/TrgSF
    events[f'weightcorr_{name}_TrgSF_singleMu_stat_{period}_Up_rel'] =    (TrgSF + ceval[HLT_name[period]].evaluate(ak.to_numpy(abs(Sel_Muon.eta)), ak.to_numpy(ev_pt), "stat"))/TrgSF
    events[f'weightcorr_{name}_TrgSF_singleMu_stat_{period}_Down_rel'] =  (TrgSF - ceval[HLT_name[period]].evaluate(ak.to_numpy(abs(Sel_Muon.eta)), ak.to_numpy(ev_pt), "stat"))/TrgSF
    return TrgSF

def get_trigger_correction_e(Sel_Electron, events, name, period):
    HLT_name = {
        '2018': 'Ele32',
        '2017': 'Ele32',
        '2016': 'Ele25',
        '2016_HIPM': 'Ele25',
    }
    # sf for e trigger from previous analysis: https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgHLTScaleFactorMeasurements
    fTrigger_path = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/{period}/electron/sf_el_{period}_HLT{HLT_name[period]}_witherr.root' 
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
    corr = ceval["tau_energy_scale"].evaluate(ak.to_numpy(flat_lep.pt), ak.to_numpy(flat_lep.eta), ak.to_numpy(flat_lep.decayMode), ak.to_numpy(flat_lep.genPartFlav), "DeepTau2017v2p1", 'nom')
    corr_up = ceval["tau_energy_scale"].evaluate(ak.to_numpy(flat_lep.pt), ak.to_numpy(flat_lep.eta), ak.to_numpy(flat_lep.decayMode), ak.to_numpy(flat_lep.genPartFlav), "DeepTau2017v2p1", 'up')
    corr_down = ceval["tau_energy_scale"].evaluate(ak.to_numpy(flat_lep.pt), ak.to_numpy(flat_lep.eta), ak.to_numpy(flat_lep.decayMode), ak.to_numpy(flat_lep.genPartFlav), "DeepTau2017v2p1", 'down')
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
    corr = ceval[PU_key[period]].evaluate(ak.to_numpy(events.Pileup.nTrueInt), 'nominal')
    corr_up = ceval[PU_key[period]].evaluate(ak.to_numpy(events.Pileup.nTrueInt), 'up')
    corr_down = ceval[PU_key[period]].evaluate(ak.to_numpy(events.Pileup.nTrueInt), 'down')

    if save_updown:
        events[f'weightcorr_PileUp_Total_Central'] = corr
        events[f'weightcorr_PileUp_Up_rel'] = corr_up/corr
        events[f'weightcorr_PileUp_Down_rel'] = corr_down/corr
    return corr, corr_up, corr_down

def compute_sf_L1PreFiring(events):
    sf_L1PreFiring = ak.to_numpy(events['L1PreFiringWeight']['Nom'])
    events['weightcorr_L1PreFiring_Total_Central'] = sf_L1PreFiring
    #events['weightcorr_L1PreFiring_Total_Up'] = ak.to_numpy(events['L1PreFiringWeight']['Up'])
    #events['weightcorr_L1PreFiring_Total_Down'] = ak.to_numpy(events['L1PreFiringWeight']['Dn'])
    events['weightcorr_L1PreFiring_ECAL_Central_rel'] = ak.to_numpy(events['L1PreFiringWeight']['ECAL_Nom'])/sf_L1PreFiring
    events['weightcorr_L1PreFiring_ECALUp_rel'] = ak.to_numpy(events['L1PreFiringWeight']['ECAL_Up'])/sf_L1PreFiring
    events['weightcorr_L1PreFiring_ECALDown_rel'] = ak.to_numpy(events['L1PreFiringWeight']['ECAL_Dn'])/sf_L1PreFiring
    events['weightcorr_L1PreFiring_Muon_Central_rel'] = ak.to_numpy(events['L1PreFiringWeight']['Muon_Nom'])/sf_L1PreFiring
    events['weightcorr_L1PreFiring_Muon_StatUp_rel'] = ak.to_numpy(events['L1PreFiringWeight']['Muon_StatUp'])/sf_L1PreFiring
    events['weightcorr_L1PreFiring_Muon_StatDown_rel'] = ak.to_numpy(events['L1PreFiringWeight']['Muon_StatDn'])/sf_L1PreFiring
    events['weightcorr_L1PreFiring_Muon_SystUp_rel'] = ak.to_numpy(events['L1PreFiringWeight']['Muon_SystUp'])/sf_L1PreFiring
    events['weightcorr_L1PreFiring_Muon_SystDown_rel'] = ak.to_numpy(events['L1PreFiringWeight']['Muon_SystDn'])/sf_L1PreFiring
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
    #effMC_err = {}
    flavors = [0,4,5]
    systematics = ["central", "up_uncorrelated", "up_correlated", "down_uncorrelated", "down_correlated"]
    for flavor in flavors:
        mask_JetFlavor[flavor] = events.bjets.hadronFlavour == flavor
        f_path = os.path.join(os.getenv("ANALYSIS_PATH"),f'CoffeaAnalysis/corrections/BTagSF/corr_schema/{period}/btagEff_{str(flavor)}.json')
        ceval = correctionlib.CorrectionSet.from_file(f_path)
        # need to do that because jagged array
        wrap_ceval = correctionlib_wrapper(ceval['BTagEff'])
        effMC[flavor] = wrap_ceval(events.bjets.pt, abs(events.bjets.eta))
        effData[flavor] = {}
        for systematic in systematics:
            effData[flavor][systematic] = {}
            effData[flavor][systematic]['PassLoose'] = myJetSF(events.bjets[mask_JetFlavor[flavor] & mask_JetPassLoose], cset[key_effData[flavor]], systematic)
            effData[flavor][systematic]['NotPassLoose'] = myJetSF(events.bjets[mask_JetFlavor[flavor] & ~mask_JetPassLoose], cset[key_effData[flavor]], systematic)
        
        #f_path_err = os.path.join(os.getenv("ANALYSIS_PATH"),f'BTagSF/corr_schema/{period}/btagEff_{str(flavor)}_err.json')
        #ceval_err = correctionlib.CorrectionSet.from_file(f_path_err)
        #effMC_err[flavor] = ceval_err['BTagEff_err'].evaluate(ak.to_numpy(events.bjets.pt), np.abs(ak.to_numpy(events.bjets.eta)))

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


def weightcorr_TauID_genuineElectron(events, Sel_Tau, name, ceval, period, DeepTauVersion):
    sf_Vse = ak.to_numpy(ceval[f"{DeepTauVersion}VSe"].evaluate(ak.to_numpy(Sel_Tau.eta), ak.to_numpy(Sel_Tau.genPartFlav), 'VVLoose', 'nom'))
    sf_Vse_up_rel = ak.to_numpy(ceval[f"{DeepTauVersion}VSe"].evaluate(ak.to_numpy(Sel_Tau.eta), ak.to_numpy(Sel_Tau.genPartFlav), 'VVLoose', 'up'))/sf_Vse
    sf_Vse_down_rel = ak.to_numpy(ceval[f"{DeepTauVersion}VSe"].evaluate(ak.to_numpy(Sel_Tau.eta), ak.to_numpy(Sel_Tau.genPartFlav), 'VVLoose', 'down'))/sf_Vse
    #SFs are eta dependent (split into barrel and endcap regions, i.e [0,1.5,2.3]), and uncorelated across area
    Sel_Tau_barrel = np.abs(Sel_Tau.eta) < 1.5
    Sel_Tau_endcaps = np.abs(Sel_Tau.eta) >= 1.5
    events[f'weightcorr_{name}_TauID_genuineElectron_{period}_barrel_Up_rel'] = ak.concatenate([ak.unflatten(sf_Vse_up_rel[Sel_Tau_barrel], Sel_Tau_barrel*1), 
                                                                                              ak.unflatten(np.ones(len(Sel_Tau_endcaps))[Sel_Tau_endcaps], Sel_Tau_endcaps*1)
                                                                                              ], axis=1)
    events[f'weightcorr_{name}_TauID_genuineElectron_{period}_barrel_Down_rel'] = ak.concatenate([ak.unflatten(sf_Vse_down_rel[Sel_Tau_barrel], Sel_Tau_barrel*1), 
                                                                                                ak.unflatten(np.ones(len(Sel_Tau_endcaps))[Sel_Tau_endcaps], Sel_Tau_endcaps*1)
                                                                                                ], axis=1)
    events[f'weightcorr_{name}_TauID_genuineElectron_{period}_endcaps_Up_rel'] = ak.concatenate([ak.unflatten(np.ones(len(Sel_Tau_barrel))[Sel_Tau_barrel], Sel_Tau_barrel*1), 
                                                                                               ak.unflatten(sf_Vse_up_rel[Sel_Tau_endcaps], Sel_Tau_endcaps*1)
                                                                                               ], axis=1) 
    events[f'weightcorr_{name}_TauID_genuineElectron_{period}_endcaps_Down_rel'] = ak.concatenate([ak.unflatten(np.ones(len(Sel_Tau_barrel))[Sel_Tau_barrel], Sel_Tau_barrel*1), 
                                                                                                 ak.unflatten(sf_Vse_down_rel[Sel_Tau_endcaps], Sel_Tau_endcaps*1)
                                                                                                 ], axis=1) 
    #events[f'weightcorr_{name}_TauID_genuineElectron_Total_Up'] =sf_Vse_up_rel*sf_Vse
    #events[f'weightcorr_{name}_TauID_genuineElectron_Total_Down'] = sf_Vse_down_rel*sf_Vse
    events[f'weightcorr_{name}_TauID_genuineElectron_Total_Central'] =sf_Vse
    return sf_Vse

def weightcorr_TauID_genuineMuon(events, Sel_Tau, name, ceval, period, DeepTauVersion):
    sf_Vsmu = ak.to_numpy(ceval[f"{DeepTauVersion}VSmu"].evaluate(ak.to_numpy(Sel_Tau.eta), ak.to_numpy(Sel_Tau.genPartFlav), 'Tight', 'nom'))
    sf_Vsmu_up_rel = ak.to_numpy(ceval[f"{DeepTauVersion}VSmu"].evaluate(ak.to_numpy(Sel_Tau.eta), ak.to_numpy(Sel_Tau.genPartFlav), 'Tight', 'up'))/sf_Vsmu
    sf_Vsmu_down_rel = ak.to_numpy(ceval[f"{DeepTauVersion}VSmu"].evaluate(ak.to_numpy(Sel_Tau.eta), ak.to_numpy(Sel_Tau.genPartFlav), 'Tight', 'down'))/sf_Vsmu
    #SFs are eta dependent (bins are [0,0.4,0.8,1.2,1.7,2.3]), and uncorelated across area 
    Sel_Tau_etaLt0p4 = np.abs(Sel_Tau.eta) < 0.4
    Sel_Tau_eta0p4to0p8 = (np.abs(Sel_Tau.eta) >= 0.4) & (np.abs(Sel_Tau.eta) < 0.8)
    Sel_Tau_eta0p8to1p2 = (np.abs(Sel_Tau.eta) >= 0.8) & (np.abs(Sel_Tau.eta) < 1.2)
    Sel_Tau_eta1p2to1p7 = (np.abs(Sel_Tau.eta) >= 1.2) & (np.abs(Sel_Tau.eta) < 1.7)
    Sel_Tau_etaGt1p7 = np.abs(Sel_Tau.eta) >= 1.7

    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_etaLt0p4_Up_rel'] = ak.concatenate([ak.unflatten(sf_Vsmu_up_rel[Sel_Tau_etaLt0p4], Sel_Tau_etaLt0p4*1), 
                    ak.unflatten(np.ones(len(Sel_Tau_eta0p4to0p8))[Sel_Tau_eta0p4to0p8], Sel_Tau_eta0p4to0p8*1), 
                    ak.unflatten(np.ones(len(Sel_Tau_eta0p8to1p2))[Sel_Tau_eta0p8to1p2], Sel_Tau_eta0p8to1p2*1),
                    ak.unflatten(np.ones(len(Sel_Tau_eta1p2to1p7))[Sel_Tau_eta1p2to1p7], Sel_Tau_eta1p2to1p7*1),
                    ak.unflatten(np.ones(len(Sel_Tau_etaGt1p7))[Sel_Tau_etaGt1p7], Sel_Tau_etaGt1p7*1)
                ], axis=1)
    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_etaLt0p4_Down_rel'] = ak.concatenate([ak.unflatten(sf_Vsmu_down_rel[Sel_Tau_etaLt0p4], Sel_Tau_etaLt0p4*1), 
                    ak.unflatten(np.ones(len(Sel_Tau_eta0p4to0p8))[Sel_Tau_eta0p4to0p8], Sel_Tau_eta0p4to0p8*1), 
                    ak.unflatten(np.ones(len(Sel_Tau_eta0p8to1p2))[Sel_Tau_eta0p8to1p2], Sel_Tau_eta0p8to1p2*1),
                    ak.unflatten(np.ones(len(Sel_Tau_eta1p2to1p7))[Sel_Tau_eta1p2to1p7], Sel_Tau_eta1p2to1p7*1),
                    ak.unflatten(np.ones(len(Sel_Tau_etaGt1p7))[Sel_Tau_etaGt1p7], Sel_Tau_etaGt1p7*1)
                ], axis=1)
    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_eta0p4to0p8_Up_rel'] = ak.concatenate([ak.unflatten(np.ones(len(Sel_Tau_etaLt0p4))[Sel_Tau_etaLt0p4], Sel_Tau_etaLt0p4*1), 
                    ak.unflatten(sf_Vsmu_up_rel[Sel_Tau_eta0p4to0p8], Sel_Tau_eta0p4to0p8*1), 
                    ak.unflatten(np.ones(len(Sel_Tau_eta0p8to1p2))[Sel_Tau_eta0p8to1p2], Sel_Tau_eta0p8to1p2*1),
                    ak.unflatten(np.ones(len(Sel_Tau_eta1p2to1p7))[Sel_Tau_eta1p2to1p7], Sel_Tau_eta1p2to1p7*1),
                    ak.unflatten(np.ones(len(Sel_Tau_etaGt1p7))[Sel_Tau_etaGt1p7], Sel_Tau_etaGt1p7*1)
                ], axis=1)
    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_eta0p4to0p8_Down_rel'] = ak.concatenate([ak.unflatten(np.ones(len(Sel_Tau_etaLt0p4))[Sel_Tau_etaLt0p4], Sel_Tau_etaLt0p4*1), 
                    ak.unflatten(sf_Vsmu_down_rel[Sel_Tau_eta0p4to0p8], Sel_Tau_eta0p4to0p8*1), 
                    ak.unflatten(np.ones(len(Sel_Tau_eta0p8to1p2))[Sel_Tau_eta0p8to1p2], Sel_Tau_eta0p8to1p2*1),
                    ak.unflatten(np.ones(len(Sel_Tau_eta1p2to1p7))[Sel_Tau_eta1p2to1p7], Sel_Tau_eta1p2to1p7*1),
                    ak.unflatten(np.ones(len(Sel_Tau_etaGt1p7))[Sel_Tau_etaGt1p7], Sel_Tau_etaGt1p7*1)
                ], axis=1)
    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_eta0p8to1p2_Up_rel'] = ak.concatenate([ak.unflatten(np.ones(len(Sel_Tau_etaLt0p4))[Sel_Tau_etaLt0p4], Sel_Tau_etaLt0p4*1), 
                    ak.unflatten(np.ones(len(Sel_Tau_eta0p4to0p8))[Sel_Tau_eta0p4to0p8], Sel_Tau_eta0p4to0p8*1), 
                    ak.unflatten(sf_Vsmu_up_rel[Sel_Tau_eta0p8to1p2], Sel_Tau_eta0p8to1p2*1),
                    ak.unflatten(np.ones(len(Sel_Tau_eta1p2to1p7))[Sel_Tau_eta1p2to1p7], Sel_Tau_eta1p2to1p7*1),
                    ak.unflatten(np.ones(len(Sel_Tau_etaGt1p7))[Sel_Tau_etaGt1p7], Sel_Tau_etaGt1p7*1)
                ], axis=1)
    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_eta0p8to1p2_Down_rel'] = ak.concatenate([ak.unflatten(np.ones(len(Sel_Tau_etaLt0p4))[Sel_Tau_etaLt0p4], Sel_Tau_etaLt0p4*1), 
                    ak.unflatten(np.ones(len(Sel_Tau_eta0p4to0p8))[Sel_Tau_eta0p4to0p8], Sel_Tau_eta0p4to0p8*1), 
                    ak.unflatten(sf_Vsmu_down_rel[Sel_Tau_eta0p8to1p2], Sel_Tau_eta0p8to1p2*1),
                    ak.unflatten(np.ones(len(Sel_Tau_eta1p2to1p7))[Sel_Tau_eta1p2to1p7], Sel_Tau_eta1p2to1p7*1),
                    ak.unflatten(np.ones(len(Sel_Tau_etaGt1p7))[Sel_Tau_etaGt1p7], Sel_Tau_etaGt1p7*1)
                ], axis=1)
    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_eta1p2to1p7_Up_rel'] = ak.concatenate([ak.unflatten(np.ones(len(Sel_Tau_etaLt0p4))[Sel_Tau_etaLt0p4], Sel_Tau_etaLt0p4*1), 
                    ak.unflatten(np.ones(len(Sel_Tau_eta0p4to0p8))[Sel_Tau_eta0p4to0p8], Sel_Tau_eta0p4to0p8*1), 
                    ak.unflatten(np.ones(len(Sel_Tau_eta0p8to1p2))[Sel_Tau_eta0p8to1p2], Sel_Tau_eta0p8to1p2*1),
                    ak.unflatten(sf_Vsmu_up_rel[Sel_Tau_eta1p2to1p7], Sel_Tau_eta1p2to1p7*1),
                    ak.unflatten(np.ones(len(Sel_Tau_etaGt1p7))[Sel_Tau_etaGt1p7], Sel_Tau_etaGt1p7*1)
                ], axis=1)
    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_eta1p2to1p7_Down_rel'] = ak.concatenate([ak.unflatten(np.ones(len(Sel_Tau_etaLt0p4))[Sel_Tau_etaLt0p4], Sel_Tau_etaLt0p4*1), 
                    ak.unflatten(np.ones(len(Sel_Tau_eta0p4to0p8))[Sel_Tau_eta0p4to0p8], Sel_Tau_eta0p4to0p8*1), 
                    ak.unflatten(np.ones(len(Sel_Tau_eta0p8to1p2))[Sel_Tau_eta0p8to1p2], Sel_Tau_eta0p8to1p2*1),
                    ak.unflatten(sf_Vsmu_down_rel[Sel_Tau_eta1p2to1p7], Sel_Tau_eta1p2to1p7*1),
                    ak.unflatten(np.ones(len(Sel_Tau_etaGt1p7))[Sel_Tau_etaGt1p7], Sel_Tau_etaGt1p7*1)
                ], axis=1)
    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_etaGt1p7_Up_rel'] = ak.concatenate([ak.unflatten(np.ones(len(Sel_Tau_etaLt0p4))[Sel_Tau_etaLt0p4], Sel_Tau_etaLt0p4*1), 
                    ak.unflatten(np.ones(len(Sel_Tau_eta0p4to0p8))[Sel_Tau_eta0p4to0p8], Sel_Tau_eta0p4to0p8*1), 
                    ak.unflatten(np.ones(len(Sel_Tau_eta0p8to1p2))[Sel_Tau_eta0p8to1p2], Sel_Tau_eta0p8to1p2*1),
                    ak.unflatten(np.ones(len(Sel_Tau_eta1p2to1p7))[Sel_Tau_eta1p2to1p7], Sel_Tau_eta1p2to1p7*1),
                    ak.unflatten(sf_Vsmu_up_rel[Sel_Tau_etaGt1p7], Sel_Tau_etaGt1p7*1)
                ], axis=1)
    events[f'weightcorr_{name}_TauID_genuineMuon_{period}_etaGt1p7_Down_rel'] = ak.concatenate([ak.unflatten(np.ones(len(Sel_Tau_etaLt0p4))[Sel_Tau_etaLt0p4], Sel_Tau_etaLt0p4*1), 
                    ak.unflatten(np.ones(len(Sel_Tau_eta0p4to0p8))[Sel_Tau_eta0p4to0p8], Sel_Tau_eta0p4to0p8*1), 
                    ak.unflatten(np.ones(len(Sel_Tau_eta0p8to1p2))[Sel_Tau_eta0p8to1p2], Sel_Tau_eta0p8to1p2*1),
                    ak.unflatten(np.ones(len(Sel_Tau_eta1p2to1p7))[Sel_Tau_eta1p2to1p7], Sel_Tau_eta1p2to1p7*1),
                    ak.unflatten(sf_Vsmu_down_rel[Sel_Tau_etaGt1p7], Sel_Tau_etaGt1p7*1)
                ], axis=1)
    #events[f'weightcorr_{name}_TauID_genuineMuon_Total_Up'] =sf_Vsmu_up_rel*sf_Vsmu
    #events[f'weightcorr_{name}_TauID_genuineMuon_Total_Down'] = sf_Vsmu_down_rel*sf_Vsmu
    events[f'weightcorr_{name}_TauID_genuineMuon_Total_Central'] =sf_Vsmu
    return sf_Vsmu

def weightcorr_TauID_genuineTau(events, Sel_Tau, name, ceval, period, DeepTauVersion):
    sf_VsJet = ak.to_numpy(ceval[f"{DeepTauVersion}VSjet"].evaluate(ak.to_numpy(Sel_Tau.pt), ak.to_numpy(Sel_Tau.decayMode), ak.to_numpy(Sel_Tau.genPartFlav), 'Medium', 'VVLoose', 'nom', 'dm'))
    # SFs from the uncertainties on the linear fit parameters
    #   - Uncstat1, Uncstat2 = statistical uncertainties (Uncstat{i}Fit) decorrelated by DM (binned in 0, 1, 10, and 11) and era
    DecayModes = ['0', '1', '10', '11']
    stat_sf_list = ['stat1', 'stat2']
    for stat in stat_sf_list:
        for dm in DecayModes:
            sf_VsJet_Up_rel = ak.to_numpy(ceval[f"{DeepTauVersion}VSjet"].evaluate(ak.to_numpy(Sel_Tau.pt), ak.to_numpy(Sel_Tau.decayMode), ak.to_numpy(Sel_Tau.genPartFlav), 'Medium', 'VVLoose', f'{stat}_dm{dm}_up', 'dm'))/sf_VsJet
            sf_VsJet_Down_rel = ak.to_numpy(ceval[f"{DeepTauVersion}VSjet"].evaluate(ak.to_numpy(Sel_Tau.pt), ak.to_numpy(Sel_Tau.decayMode), ak.to_numpy(Sel_Tau.genPartFlav), 'Medium', 'VVLoose', f'{stat}_dm{dm}_down', 'dm'))/sf_VsJet
            list_up = []
            list_down = []
            for decayM in DecayModes:
                Sel_Tau_dm = Sel_Tau.decayMode == int(decayM)
                if decayM == dm:
                    list_up.append(ak.unflatten(sf_VsJet_Up_rel[Sel_Tau_dm], Sel_Tau_dm*1))
                    list_down.append(ak.unflatten(sf_VsJet_Down_rel[Sel_Tau_dm], Sel_Tau_dm*1))
                else:
                    list_up.append(ak.unflatten(np.ones(len(Sel_Tau_dm))[Sel_Tau_dm], Sel_Tau_dm*1))
                    list_down.append(ak.unflatten(np.ones(len(Sel_Tau_dm))[Sel_Tau_dm], Sel_Tau_dm*1))
            List_UP_conc = ak.concatenate(list_up, axis=1)
            List_down_conc = ak.concatenate(list_down, axis=1)

            List_UP_null_mask = ak.num(List_UP_conc) == 0
            if ak.sum(List_UP_null_mask) != 0:
                print(f'issue with weightcorr_{name}_TauID_genuineTau_Unc{stat}_DM{dm}_{period}_Up_rel')
                List_UP_conc = ak.where(List_UP_null_mask, ak.Array([[1]]), List_UP_conc)
                #List_UP_conc = ak.to_regular(List_UP_conc)

            List_down_null_mask = ak.num(List_down_conc) == 0
            if ak.sum(List_down_null_mask) != 0:
                print(f'issue with weightcorr_{name}_TauID_genuineTau_Unc{stat}_DM{dm}_{period}_Down_rel')
                List_down_conc = ak.where(List_down_null_mask, ak.Array([[1]]), List_down_conc)
                #List_down_conc = ak.to_regular(List_down_conc)

            events[f'weightcorr_{name}_TauID_genuineTau_Unc{stat}_DM{dm}_{period}_Up_rel'] = List_UP_conc
            events[f'weightcorr_{name}_TauID_genuineTau_Unc{stat}_DM{dm}_{period}_Down_rel'] = List_down_conc

    #   - UncSystAllEras = The component of the systematic uncertainty that is correlated across DMs and eras
    events[f'weightcorr_{name}_TauID_genuineTau_UncSystAllEras_Up_rel'] = ak.to_numpy(ceval[f"{DeepTauVersion}VSjet"].evaluate(ak.to_numpy(Sel_Tau.pt), ak.to_numpy(Sel_Tau.decayMode), ak.to_numpy(Sel_Tau.genPartFlav), 'Medium', 'VVLoose', 'syst_alleras_up', 'dm'))/sf_VsJet
    events[f'weightcorr_{name}_TauID_genuineTau_UncSystAllEras_Down_rel'] = ak.to_numpy(ceval[f"{DeepTauVersion}VSjet"].evaluate(ak.to_numpy(Sel_Tau.pt), ak.to_numpy(Sel_Tau.decayMode), ak.to_numpy(Sel_Tau.genPartFlav), 'Medium', 'VVLoose', 'syst_alleras_down', 'dm'))/sf_VsJet

    #   - UncSyst{Era} = The component of the systematic uncertainty that is correlated across DMs but uncorrelated by eras
    events[f'weightcorr_{name}_TauID_genuineTau_UncSyst_{period}_Up_rel'] = ak.to_numpy(ceval[f"{DeepTauVersion}VSjet"].evaluate(ak.to_numpy(Sel_Tau.pt), ak.to_numpy(Sel_Tau.decayMode), ak.to_numpy(Sel_Tau.genPartFlav), 'Medium', 'VVLoose', f'syst_{Area_ref[period]}_up', 'dm'))/sf_VsJet
    events[f'weightcorr_{name}_TauID_genuineTau_UncSyst_{period}_Down_rel'] = ak.to_numpy(ceval[f"{DeepTauVersion}VSjet"].evaluate(ak.to_numpy(Sel_Tau.pt), ak.to_numpy(Sel_Tau.decayMode), ak.to_numpy(Sel_Tau.genPartFlav), 'Medium', 'VVLoose', f'syst_{Area_ref[period]}_down', 'dm'))/sf_VsJet

    if DeepTauVersion == "DeepTau2018v2p5":
        #   - UncSyst{Era}_DM{DM} = The component of the systematic uncertainty due to the tau energy scale that is correlated across DMs and eras
        for dm in DecayModes:
            sf_VsJet_Up_rel = ak.to_numpy(ceval["DeepTau2018v2p5VSjet"].evaluate(ak.to_numpy(Sel_Tau.pt), ak.to_numpy(Sel_Tau.decayMode), ak.to_numpy(Sel_Tau.genPartFlav), 'Medium', 'VVLoose', f'syst_TES_{period}_dm{dm}_up', 'dm'))/sf_VsJet
            sf_VsJet_Down_rel = ak.to_numpy(ceval["DeepTau2018v2p5VSjet"].evaluate(ak.to_numpy(Sel_Tau.pt), ak.to_numpy(Sel_Tau.decayMode), ak.to_numpy(Sel_Tau.genPartFlav), 'Medium', 'VVLoose', f'syst_TES_{period}_dm{dm}_down', 'dm'))/sf_VsJet
            list_up = []
            list_down = []
            for decayM in DecayModes:
                Sel_Tau_dm = Sel_Tau.decayMode == int(decayM)
                if decayM == dm:
                    list_up.append(ak.unflatten(sf_VsJet_Up_rel[Sel_Tau_dm], Sel_Tau_dm*1))
                    list_down.append(ak.unflatten(sf_VsJet_Down_rel[Sel_Tau_dm], Sel_Tau_dm*1))
                else:
                    list_up.append(ak.unflatten(np.ones(len(Sel_Tau_dm))[Sel_Tau_dm], Sel_Tau_dm*1))
                    list_down.append(ak.unflatten(np.ones(len(Sel_Tau_dm))[Sel_Tau_dm], Sel_Tau_dm*1))
            List_UP_conc = ak.concatenate(list_up, axis=1)
            List_down_conc = ak.concatenate(list_down, axis=1)

            List_UP_null_mask = ak.num(List_UP_conc) == 0
            if ak.sum(List_UP_null_mask) != 0:
                print(f'issue with weightcorr_{name}_TauID_genuineTau_UncSyst_DM{dm}_{period}_Up_rel')
                List_UP_conc = ak.where(List_UP_null_mask, ak.Array([[1]]), List_UP_conc)
                #List_UP_conc = ak.to_regular(List_UP_conc)

            List_down_null_mask = ak.num(List_down_conc) == 0
            if ak.sum(List_down_null_mask) != 0:
                print(f'issue with weightcorr_{name}_TauID_genuineTau_UncSyst_DM{dm}_{period}_Down_rel')
                List_down_conc = ak.where(List_down_null_mask, ak.Array([[1]]), List_down_conc)
                #List_down_conc = ak.to_regular(List_down_conc)

            events[f'weightcorr_{name}_TauID_genuineTau_UncSyst_DM{dm}_{period}_Up_rel'] = List_UP_conc
            events[f'weightcorr_{name}_TauID_genuineTau_UncSyst_DM{dm}_{period}_Down_rel'] =List_down_conc

    if DeepTauVersion == "DeepTau2017v2p1":
        #   - UncSyst{Era}_DM{DM} = The component of the systematic uncertainty that is correlated across DMs and eras
        for dm in DecayModes:
            sf_VsJet_Up_rel = ak.to_numpy(ceval["DeepTau2017v2p1VSjet"].evaluate(ak.to_numpy(Sel_Tau.pt), ak.to_numpy(Sel_Tau.decayMode), ak.to_numpy(Sel_Tau.genPartFlav), 'Medium', 'VVLoose', f'syst_dm{dm}_{Area_ref[period]}_up', 'dm'))/sf_VsJet
            sf_VsJet_Down_rel = ak.to_numpy(ceval["DeepTau2017v2p1VSjet"].evaluate(ak.to_numpy(Sel_Tau.pt), ak.to_numpy(Sel_Tau.decayMode), ak.to_numpy(Sel_Tau.genPartFlav), 'Medium', 'VVLoose', f'syst_dm{dm}_{Area_ref[period]}_down', 'dm'))/sf_VsJet
            list_up = []
            list_down = []
            for decayM in DecayModes:
                Sel_Tau_dm = Sel_Tau.decayMode == int(decayM)
                if decayM == dm:
                    list_up.append(ak.unflatten(sf_VsJet_Up_rel[Sel_Tau_dm], Sel_Tau_dm*1))
                    list_down.append(ak.unflatten(sf_VsJet_Down_rel[Sel_Tau_dm], Sel_Tau_dm*1))
                else:
                    list_up.append(ak.unflatten(np.ones(len(Sel_Tau_dm))[Sel_Tau_dm], Sel_Tau_dm*1))
                    list_down.append(ak.unflatten(np.ones(len(Sel_Tau_dm))[Sel_Tau_dm], Sel_Tau_dm*1))
            events[f'weightcorr_{name}_TauID_genuineTau_UncSyst_DM{dm}_{period}_Up_rel'] = ak.concatenate(list_up, axis=1)
            events[f'weightcorr_{name}_TauID_genuineTau_UncSyst_DM{dm}_{period}_Down_rel'] =ak.concatenate(list_down, axis=1)
            List_UP_conc = ak.concatenate(list_up, axis=1)
            List_down_conc = ak.concatenate(list_down, axis=1)

            List_UP_null_mask = ak.num(List_UP_conc) == 0
            if ak.sum(List_UP_null_mask) != 0:
                print(f'issue with weightcorr_{name}_TauID_genuineTau_UncSyst_DM{dm}_{period}_Up_rel')
                List_UP_conc = ak.where(List_UP_null_mask, ak.Array([[1]]), List_UP_conc)
                #List_UP_conc = ak.to_regular(List_UP_conc)

            List_down_null_mask = ak.num(List_down_conc) == 0
            if ak.sum(List_down_null_mask) != 0:
                print(f'issue with weightcorr_{name}_TauID_genuineTau_UncSyst_DM{dm}_{period}_Down_rel')
                List_down_conc = ak.where(List_down_null_mask, ak.Array([[1]]), List_down_conc)
                #List_down_conc = ak.to_regular(List_down_conc)

            events[f'weightcorr_{name}_TauID_genuineTau_UncSyst_DM{dm}_{period}_Up_rel'] = List_UP_conc
            events[f'weightcorr_{name}_TauID_genuineTau_UncSyst_DM{dm}_{period}_Down_rel'] =List_down_conc

    #events[f'weightcorr_{name}_TauID_genuineTau_Total_Up'] = ak.to_numpy(ceval[f"{DeepTauVersion}VSjet"].evaluate(ak.to_numpy(Sel_Tau.pt), ak.to_numpy(Sel_Tau.decayMode), ak.to_numpy(Sel_Tau.genPartFlav), 'Medium', 'VVLoose', 'up', 'dm'))
    #events[f'weightcorr_{name}_TauID_genuineTau_Total_Down'] = ak.to_numpy(ceval[f"{DeepTauVersion}VSjet"].evaluate(ak.to_numpy(Sel_Tau.pt), ak.to_numpy(Sel_Tau.decayMode), ak.to_numpy(Sel_Tau.genPartFlav), 'Medium', 'VVLoose', 'down', 'dm'))
    events[f'weightcorr_{name}_TauID_genuineTau_Total_Central'] = sf_VsJet
    return sf_VsJet

def get_muonsf_Below20(events, Sel_Muon, name, ceval, period):
    # sf for muon with pt<20 GeV
    nlepBelow20 = Sel_Muon.pt <= 20
    #RECO
    # pt binning is [40, inf] but the recommendation is to apply them for muons with pT in the range [10; 200] GeV.
    ev_pt = ak.to_numpy(Sel_Muon.pt)
    ev_pt = np.where(ev_pt<40, 40, ev_pt)
    RECOsf_Below20 = ceval['NUM_TrackerMuons_DEN_genTracks'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepBelow20])), ak.to_numpy(ev_pt[nlepBelow20]), "nominal")
    #ID
    IDevaluator_Below20 = get_scales_fromjson(f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/'+period+'/muon/Efficiency_muon_trackerMuon_Run' + Area_dir[period] + '_ID.json') # custom tab from twiki because ID sf in muon_Z.json are for Muon.pt > 15
    IDsf_Below20 = ak.to_numpy(IDevaluator_Below20["NUM_MediumID_DEN_TrackerMuons/abseta_pt_value"](abs(Sel_Muon.eta[nlepBelow20]), Sel_Muon.pt[nlepBelow20])) 
    #total
    sf_Below20 = ak.from_numpy(RECOsf_Below20*IDsf_Below20)

    #save sf variation Below20
    #RECO
    RECOsf_Below20_Up_rel = ceval['NUM_TrackerMuons_DEN_genTracks'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepBelow20])), ak.to_numpy(ev_pt[nlepBelow20]), "systup")/sf_Below20
    RECOsf_Below20_Down_rel = ceval['NUM_TrackerMuons_DEN_genTracks'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepBelow20])), ak.to_numpy(ev_pt[nlepBelow20]), "systdown")/sf_Below20
    events[f'weightcorr_{name}_MuID_RECO_ptlt20_Up_rel'] = ak.concatenate([ak.unflatten(RECOsf_Below20_Up_rel*IDsf_Below20, nlepBelow20*1),
                                                                             ak.unflatten(np.ones(len(~nlepBelow20))[~nlepBelow20], ~nlepBelow20*1), 
                                                                             ], axis=1)
    events[f'weightcorr_{name}_MuID_RECO_ptlt20_Down_rel'] = ak.concatenate([ak.unflatten(RECOsf_Below20_Down_rel*IDsf_Below20, nlepBelow20*1),
                                                                             ak.unflatten(np.ones(len(~nlepBelow20))[~nlepBelow20], ~nlepBelow20*1), 
                                                                             ], axis=1)
    #ID
    IDsf_Below20_err = IDevaluator_Below20["NUM_MediumID_DEN_TrackerMuons/abseta_pt_error"](abs(Sel_Muon.eta[nlepBelow20]), Sel_Muon.pt[nlepBelow20])
    IDsf_Below20_Up_rel = ak.from_numpy((IDsf_Below20+IDsf_Below20_err))/sf_Below20
    IDsf_Below20_Down_rel = ak.from_numpy((IDsf_Below20-IDsf_Below20_err))/sf_Below20
    events[f'weightcorr_{name}_MuID_MediumID_ptlt20_Up_rel'] = ak.concatenate([ak.unflatten(IDsf_Below20_Up_rel*RECOsf_Below20, nlepBelow20*1),
                                                                             ak.unflatten(np.ones(len(~nlepBelow20))[~nlepBelow20], ~nlepBelow20*1), 
                                                                             ], axis=1)
    events[f'weightcorr_{name}_MuID_MediumID_ptlt20_Down_rel'] = ak.concatenate([ak.unflatten(IDsf_Below20_Down_rel*RECOsf_Below20, nlepBelow20*1),
                                                                             ak.unflatten(np.ones(len(~nlepBelow20))[~nlepBelow20], ~nlepBelow20*1), 
                                                                             ], axis=1)
    
    sf_Below20_Up = ak.from_numpy(RECOsf_Below20_Up_rel*sf_Below20*IDsf_Below20_Up_rel*sf_Below20)
    sf_Below20_Down = ak.from_numpy(RECOsf_Below20_Down_rel*sf_Below20*IDsf_Below20_Down_rel*sf_Below20)
    return sf_Below20, sf_Below20_Up, sf_Below20_Down

def get_muonsf_Between15and120(events, Sel_Muon, name, ceval, period):
    # sf for muon with 15<pt<120 GeV (from central repo)
    nlepBetween15and120 = (Sel_Muon.pt > 15) & (Sel_Muon.pt <= 120)
    # For RECO pt binning is [40, inf] but the recommendation is to apply them for muons with pT in the range [10; 200] GeV.
    ev_pt = ak.to_numpy(Sel_Muon.pt)
    ev_pt = np.where(ev_pt<40, 40, ev_pt)
    #RECO
    RECOsf_Between15and120 = ceval['NUM_TrackerMuons_DEN_genTracks'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepBetween15and120])), ak.to_numpy(ev_pt[nlepBetween15and120]), "nominal")
    #ID
    IDsf_Between15and120 = ceval['NUM_MediumID_DEN_TrackerMuons'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepBetween15and120])), ak.to_numpy(Sel_Muon.pt[nlepBetween15and120]), 'nominal')
    #ISO
    ISOsf_Between15and120 = ceval['NUM_LooseRelIso_DEN_MediumID'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepBetween15and120])), ak.to_numpy(Sel_Muon.pt[nlepBetween15and120]), 'nominal')
    #total
    sf_Between15and120 = ak.from_numpy(RECOsf_Between15and120*IDsf_Between15and120*ISOsf_Between15and120)

    #save sf variation Between15and120
    #--RECO--
    #syst
    RECOsf_Between15and120_syst_Up_rel = (RECOsf_Between15and120 + ceval['NUM_TrackerMuons_DEN_genTracks'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepBetween15and120])), ak.to_numpy(ev_pt[nlepBetween15and120]), "syst"))/RECOsf_Between15and120
    RECOsf_Between15and120_syst_Down_rel = (RECOsf_Between15and120 - ceval['NUM_TrackerMuons_DEN_genTracks'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepBetween15and120])), ak.to_numpy(ev_pt[nlepBetween15and120]), "syst"))/RECOsf_Between15and120
    events[f'weightcorr_{name}_MuID_RECO_pt20to120_syst_Up_rel'] = ak.concatenate([ak.unflatten(RECOsf_Between15and120_syst_Up_rel, nlepBetween15and120*1),
                                                                             ak.unflatten(np.ones(len(~nlepBetween15and120))[~nlepBetween15and120], ~nlepBetween15and120*1), 
                                                                             ], axis=1)
    events[f'weightcorr_{name}_MuID_RECO_pt20to120_syst_Down_rel'] = ak.concatenate([ak.unflatten(RECOsf_Between15and120_syst_Down_rel, nlepBetween15and120*1),
                                                                             ak.unflatten(np.ones(len(~nlepBetween15and120))[~nlepBetween15and120], ~nlepBetween15and120*1), 
                                                                             ], axis=1)
    #stat
    RECOsf_Between15and120_stat_Up_rel = (RECOsf_Between15and120 + ceval['NUM_TrackerMuons_DEN_genTracks'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepBetween15and120])), ak.to_numpy(ev_pt[nlepBetween15and120]), "stat"))/RECOsf_Between15and120
    RECOsf_Between15and120_stat_Down_rel = (RECOsf_Between15and120 - ceval['NUM_TrackerMuons_DEN_genTracks'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepBetween15and120])), ak.to_numpy(ev_pt[nlepBetween15and120]), "stat"))/RECOsf_Between15and120
    events[f'weightcorr_{name}_MuID_RECO_pt20to120_stat_{period}_Up_rel'] = ak.concatenate([ak.unflatten(RECOsf_Between15and120_stat_Up_rel, nlepBetween15and120*1),
                                                                             ak.unflatten(np.ones(len(~nlepBetween15and120))[~nlepBetween15and120], ~nlepBetween15and120*1), 
                                                                             ], axis=1)
    events[f'weightcorr_{name}_MuID_RECO_pt20to120_stat_{period}_Down_rel'] = ak.concatenate([ak.unflatten(RECOsf_Between15and120_stat_Down_rel, nlepBetween15and120*1),
                                                                             ak.unflatten(np.ones(len(~nlepBetween15and120))[~nlepBetween15and120], ~nlepBetween15and120*1), 
                                                                             ], axis=1)
    
    #--ID--
    #syst
    IDsf_Between15and120_syst_Up_rel = (IDsf_Between15and120 + ceval['NUM_MediumID_DEN_TrackerMuons'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepBetween15and120])), ak.to_numpy(Sel_Muon.pt[nlepBetween15and120]), 'syst'))/IDsf_Between15and120
    IDsf_Between15and120_syst_Down_rel = (IDsf_Between15and120 - ceval['NUM_MediumID_DEN_TrackerMuons'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepBetween15and120])), ak.to_numpy(Sel_Muon.pt[nlepBetween15and120]), 'syst'))/IDsf_Between15and120
    events[f'weightcorr_{name}_MuID_MediumID_pt20to120_syst_Up_rel'] = ak.concatenate([ak.unflatten(IDsf_Between15and120_syst_Up_rel, nlepBetween15and120*1),
                                                                             ak.unflatten(np.ones(len(~nlepBetween15and120))[~nlepBetween15and120], ~nlepBetween15and120*1), 
                                                                             ], axis=1) 
    events[f'weightcorr_{name}_MuID_MediumID_pt20to120_syst_Down_rel'] = ak.concatenate([ak.unflatten(IDsf_Between15and120_syst_Down_rel, nlepBetween15and120*1),
                                                                             ak.unflatten(np.ones(len(~nlepBetween15and120))[~nlepBetween15and120], ~nlepBetween15and120*1), 
                                                                             ], axis=1) 
    #stat
    IDsf_Between15and120_stat_Up_rel = (IDsf_Between15and120 + ceval['NUM_MediumID_DEN_TrackerMuons'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepBetween15and120])), ak.to_numpy(Sel_Muon.pt[nlepBetween15and120]), 'stat'))/IDsf_Between15and120
    IDsf_Between15and120_stat_Down_rel = (IDsf_Between15and120 - ceval['NUM_MediumID_DEN_TrackerMuons'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepBetween15and120])), ak.to_numpy(Sel_Muon.pt[nlepBetween15and120]), 'stat'))/IDsf_Between15and120
    events[f'weightcorr_{name}_MuID_MediumID_pt20to120_stat_{period}_Up_rel'] = ak.concatenate([ak.unflatten(IDsf_Between15and120_stat_Up_rel, nlepBetween15and120*1),
                                                                             ak.unflatten(np.ones(len(~nlepBetween15and120))[~nlepBetween15and120], ~nlepBetween15and120*1), 
                                                                             ], axis=1) 
    events[f'weightcorr_{name}_MuID_MediumID_pt20to120_stat_{period}_Down_rel'] = ak.concatenate([ak.unflatten(IDsf_Between15and120_stat_Down_rel, nlepBetween15and120*1),
                                                                             ak.unflatten(np.ones(len(~nlepBetween15and120))[~nlepBetween15and120], ~nlepBetween15and120*1), 
                                                                             ], axis=1) 
    
    #--ISO--
    #syst
    ISOsf_Between15and120_syst_Up_rel = (ISOsf_Between15and120 + ceval['NUM_LooseRelIso_DEN_MediumID'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepBetween15and120])), ak.to_numpy(Sel_Muon.pt[nlepBetween15and120]), 'syst'))/ISOsf_Between15and120
    ISOsf_Between15and120_syst_Down_rel = (ISOsf_Between15and120 - ceval['NUM_LooseRelIso_DEN_MediumID'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepBetween15and120])), ak.to_numpy(Sel_Muon.pt[nlepBetween15and120]), 'syst'))/ISOsf_Between15and120
    events[f'weightcorr_{name}_MuID_LooseISO_pt20to120_syst_Up_rel'] = ak.concatenate([ak.unflatten(ISOsf_Between15and120_syst_Up_rel, nlepBetween15and120*1),
                                                                             ak.unflatten(np.ones(len(~nlepBetween15and120))[~nlepBetween15and120], ~nlepBetween15and120*1), 
                                                                             ], axis=1) 
    events[f'weightcorr_{name}_MuID_LooseISO_pt20to120_syst_Down_rel'] = ak.concatenate([ak.unflatten(ISOsf_Between15and120_syst_Down_rel, nlepBetween15and120*1),
                                                                             ak.unflatten(np.ones(len(~nlepBetween15and120))[~nlepBetween15and120], ~nlepBetween15and120*1), 
                                                                             ], axis=1) 
    #stat
    ISOsf_Between15and120_stat_Up_rel = (ISOsf_Between15and120 + ceval['NUM_LooseRelIso_DEN_MediumID'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepBetween15and120])), ak.to_numpy(Sel_Muon.pt[nlepBetween15and120]), 'stat'))/ISOsf_Between15and120
    ISOsf_Between15and120_stat_Down_rel = (ISOsf_Between15and120 - ceval['NUM_LooseRelIso_DEN_MediumID'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepBetween15and120])), ak.to_numpy(Sel_Muon.pt[nlepBetween15and120]), 'stat'))/ISOsf_Between15and120
    events[f'weightcorr_{name}_MuID_LooseISO_pt20to120_stat_{period}_Up_rel'] = ak.concatenate([ak.unflatten(ISOsf_Between15and120_stat_Up_rel, nlepBetween15and120*1),
                                                                             ak.unflatten(np.ones(len(~nlepBetween15and120))[~nlepBetween15and120], ~nlepBetween15and120*1), 
                                                                             ], axis=1) 
    events[f'weightcorr_{name}_MuID_LooseISO_pt20to120_stat_{period}_Down_rel'] = ak.concatenate([ak.unflatten(ISOsf_Between15and120_stat_Down_rel, nlepBetween15and120*1),
                                                                             ak.unflatten(np.ones(len(~nlepBetween15and120))[~nlepBetween15and120], ~nlepBetween15and120*1), 
                                                                             ], axis=1) 

    #sf_Between20and120_Up = ak.from_numpy(RECOsf_Between20and120_Up_rel*sf_Between20and120*IDsf_Between20and120_Up_rel*sf_Between20and120*ISOsf_Between20and120_Up_rel*sf_Between20and120)
    #sf_Between20and120_Down = ak.from_numpy(RECOsf_Between20and120_Down_rel*sf_Between20and120*IDsf_Between20and120_Down_rel*sf_Between20and120*ISOsf_Between20and120_Down_rel*sf_Between20and120)
    return sf_Between15and120 #, sf_Between20and120_Up, sf_Between20and120_Down

def get_muonsf_Above120(events, Sel_Muon, name, ceval, period):
    # sf for muon with pt>120 GeV (ID and ISO same as muon with 20<pt<120 GeV)
    nlepAbove120 = Sel_Muon.pt > 120
    #RECO
    RECOevaluator_Above120 = correctionlib.CorrectionSet.from_file(f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/'+period+f'/muon/ScaleFactors_Muon_highPt_RECO_{Area_ref[period]}_schemaV2.json')# custom tab from twiki because sf are not in muon_Z.json
    RECOsf_Above120 = RECOevaluator_Above120["NUM_GlobalMuons_DEN_TrackerMuonProbes"].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepAbove120])), ak.to_numpy(Sel_Muon.p[nlepAbove120]), 'nominal')
    #ID
    IDsf_Above120 = ceval['NUM_HighPtID_DEN_TrackerMuons'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepAbove120])), ak.to_numpy(Sel_Muon.pt[nlepAbove120]), 'nominal')
    #ISO
    ISOsf_Above120 = ceval['NUM_LooseRelIso_DEN_MediumID'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepAbove120])), ak.to_numpy(Sel_Muon.pt[nlepAbove120]), 'nominal')
    #total
    sf_Above120 = ak.from_numpy(RECOsf_Above120*IDsf_Above120*ISOsf_Above120)

    #save sf variation Above120
    #--RECO--
    #syst
    RECOsf_Above120_syst_Up_rel = (RECOsf_Above120 + RECOevaluator_Above120["NUM_GlobalMuons_DEN_TrackerMuonProbes"].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepAbove120])), ak.to_numpy(Sel_Muon.p[nlepAbove120]), 'syst'))/RECOsf_Above120
    RECOsf_Above120_syst_Down_rel = (RECOsf_Above120 - RECOevaluator_Above120["NUM_GlobalMuons_DEN_TrackerMuonProbes"].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepAbove120])), ak.to_numpy(Sel_Muon.p[nlepAbove120]), 'syst'))/RECOsf_Above120
    events[f'weightcorr_{name}_MuID_RECO_ptgt120_syst_Up_rel'] = ak.concatenate([ak.unflatten(RECOsf_Above120_syst_Up_rel, nlepAbove120*1),
                                                                             ak.unflatten(np.ones(len(~nlepAbove120))[~nlepAbove120], ~nlepAbove120*1), 
                                                                             ], axis=1) 
    events[f'weightcorr_{name}_MuID_RECO_ptgt120_syst_Down_rel'] = ak.concatenate([ak.unflatten(RECOsf_Above120_syst_Down_rel, nlepAbove120*1),
                                                                             ak.unflatten(np.ones(len(~nlepAbove120))[~nlepAbove120], ~nlepAbove120*1), 
                                                                             ], axis=1) 
    #stat
    RECOsf_Above120_stat_Up_rel = (RECOsf_Above120 + RECOevaluator_Above120["NUM_GlobalMuons_DEN_TrackerMuonProbes"].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepAbove120])), ak.to_numpy(Sel_Muon.p[nlepAbove120]), 'stat'))/RECOsf_Above120
    RECOsf_Above120_stat_Down_rel = (RECOsf_Above120 - RECOevaluator_Above120["NUM_GlobalMuons_DEN_TrackerMuonProbes"].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepAbove120])), ak.to_numpy(Sel_Muon.p[nlepAbove120]), 'stat'))/RECOsf_Above120
    events[f'weightcorr_{name}_MuID_RECO_ptgt120_stat_{period}_Up_rel'] = ak.concatenate([ak.unflatten(RECOsf_Above120_stat_Up_rel, nlepAbove120*1),
                                                                             ak.unflatten(np.ones(len(~nlepAbove120))[~nlepAbove120], ~nlepAbove120*1), 
                                                                             ], axis=1) 
    events[f'weightcorr_{name}_MuID_RECO_ptgt120_stat_{period}_Down_rel'] = ak.concatenate([ak.unflatten(RECOsf_Above120_stat_Down_rel, nlepAbove120*1),
                                                                             ak.unflatten(np.ones(len(~nlepAbove120))[~nlepAbove120], ~nlepAbove120*1), 
                                                                             ], axis=1) 

    #--ID--
    #syst
    IDsf_Above120_syst_Up_rel = (IDsf_Above120 + ceval['NUM_HighPtID_DEN_TrackerMuons'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepAbove120])), ak.to_numpy(Sel_Muon.pt[nlepAbove120]), 'syst'))/IDsf_Above120
    IDsf_Above120_syst_Down_rel = (IDsf_Above120 - ceval['NUM_HighPtID_DEN_TrackerMuons'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepAbove120])), ak.to_numpy(Sel_Muon.pt[nlepAbove120]), 'syst'))/IDsf_Above120
    events[f'weightcorr_{name}_MuID_MediumID_ptgt120_syst_Up_rel'] = ak.concatenate([ak.unflatten(IDsf_Above120_syst_Up_rel, nlepAbove120*1),
                                                                             ak.unflatten(np.ones(len(~nlepAbove120))[~nlepAbove120], ~nlepAbove120*1), 
                                                                             ], axis=1) 
    events[f'weightcorr_{name}_MuID_MediumID_ptgt120_syst_Down_rel'] = ak.concatenate([ak.unflatten(IDsf_Above120_syst_Down_rel, nlepAbove120*1),
                                                                             ak.unflatten(np.ones(len(~nlepAbove120))[~nlepAbove120], ~nlepAbove120*1), 
                                                                             ], axis=1) 
    #stat
    IDsf_Above120_stat_Up_rel = (IDsf_Above120 + ceval['NUM_HighPtID_DEN_TrackerMuons'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepAbove120])), ak.to_numpy(Sel_Muon.pt[nlepAbove120]), 'stat'))/IDsf_Above120
    IDsf_Above120_stat_Down_rel = (IDsf_Above120 - ceval['NUM_HighPtID_DEN_TrackerMuons'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepAbove120])), ak.to_numpy(Sel_Muon.pt[nlepAbove120]), 'stat'))/IDsf_Above120
    events[f'weightcorr_{name}_MuID_MediumID_ptgt120_stat_{period}_Up_rel'] = ak.concatenate([ak.unflatten(IDsf_Above120_stat_Up_rel, nlepAbove120*1),
                                                                             ak.unflatten(np.ones(len(~nlepAbove120))[~nlepAbove120], ~nlepAbove120*1), 
                                                                             ], axis=1) 
    events[f'weightcorr_{name}_MuID_MediumID_ptgt120_stat_{period}_Down_rel'] = ak.concatenate([ak.unflatten(IDsf_Above120_stat_Down_rel, nlepAbove120*1),
                                                                             ak.unflatten(np.ones(len(~nlepAbove120))[~nlepAbove120], ~nlepAbove120*1), 
                                                                             ], axis=1) 

    #--ISO--
    #syst
    ISOsf_Above120_syst_Up_rel = (ISOsf_Above120 + ceval['NUM_LooseRelIso_DEN_MediumID'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepAbove120])), ak.to_numpy(Sel_Muon.pt[nlepAbove120]), 'syst'))/ISOsf_Above120
    ISOsf_Above120_syst_Down_rel = (ISOsf_Above120 - ceval['NUM_LooseRelIso_DEN_MediumID'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepAbove120])), ak.to_numpy(Sel_Muon.pt[nlepAbove120]), 'syst'))/ISOsf_Above120
    events[f'weightcorr_{name}_MuID_LooseISO_ptgt120_syst_Up_rel'] = ak.concatenate([ak.unflatten(ISOsf_Above120_syst_Up_rel, nlepAbove120*1),
                                                                             ak.unflatten(np.ones(len(~nlepAbove120))[~nlepAbove120], ~nlepAbove120*1), 
                                                                             ], axis=1) 
    events[f'weightcorr_{name}_MuID_LooseISO_ptgt120_syst_Down_rel'] = ak.concatenate([ak.unflatten(ISOsf_Above120_syst_Down_rel, nlepAbove120*1),
                                                                             ak.unflatten(np.ones(len(~nlepAbove120))[~nlepAbove120], ~nlepAbove120*1), 
                                                                             ], axis=1) 
    #stat
    ISOsf_Above120_stat_Up_rel = (ISOsf_Above120 + ceval['NUM_LooseRelIso_DEN_MediumID'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepAbove120])), ak.to_numpy(Sel_Muon.pt[nlepAbove120]), 'stat'))/ISOsf_Above120
    ISOsf_Above120_stat_Down_rel = (ISOsf_Above120 - ceval['NUM_LooseRelIso_DEN_MediumID'].evaluate(ak.to_numpy(abs(Sel_Muon.eta[nlepAbove120])), ak.to_numpy(Sel_Muon.pt[nlepAbove120]), 'stat'))/ISOsf_Above120
    events[f'weightcorr_{name}_MuID_LooseISO_ptgt120_stat_{period}_Up_rel'] = ak.concatenate([ak.unflatten(ISOsf_Above120_stat_Up_rel, nlepAbove120*1),
                                                                             ak.unflatten(np.ones(len(~nlepAbove120))[~nlepAbove120], ~nlepAbove120*1), 
                                                                             ], axis=1) 
    events[f'weightcorr_{name}_MuID_LooseISO_ptgt120_stat_{period}_Down_rel'] = ak.concatenate([ak.unflatten(ISOsf_Above120_stat_Down_rel, nlepAbove120*1),
                                                                             ak.unflatten(np.ones(len(~nlepAbove120))[~nlepAbove120], ~nlepAbove120*1), 
                                                                             ], axis=1) 

    #sf_Above120_Up = ak.from_numpy(RECOsf_Above120_Up_rel*sf_Above120*IDsf_Above120_Up_rel*sf_Above120*ISOsf_Above120_Up_rel*sf_Above120)
    #sf_Above120_Down = ak.from_numpy(RECOsf_Above120_Down_rel*sf_Above120*IDsf_Above120_Down_rel*sf_Above120*ISOsf_Above120_Down_rel*sf_Above120)
    return sf_Above120 #, sf_Above120_Up, sf_Above120_Down

def get_electronRECOsf_Below20(events, Sel_Electron, name, period):
    #RECO sf for e
    file_name = {
        '2018': 'UL2018',
        '2017': 'UL2017',
        '2016': 'UL2016postVFP',
        '2016_HIPM': 'UL2016preVFP',
    }
    fReco_path_ptBelow20 = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/{period}/electron/egammaEffi_ptBelow20-txt_EGM2D_{file_name[period]}_witherr.root' 
    evaluator_Below20 = get_scales_fromjson(fReco_path_ptBelow20)
    mask_Below20 = Sel_Electron.pt<20
    RECOsf_Below20 = evaluator_Below20["EGamma_SF2D"](Sel_Electron.pt[mask_Below20], Sel_Electron.eta[mask_Below20])
    RECOsf_Below20_err = evaluator_Below20["EGamma_SF2D_err"](Sel_Electron.pt[mask_Below20], Sel_Electron.eta[mask_Below20])
    RECOsf_Below20_Up = RECOsf_Below20 + RECOsf_Below20_err
    RECOsf_Below20_Down = RECOsf_Below20 - RECOsf_Below20_err
    return RECOsf_Below20, RECOsf_Below20_Up, RECOsf_Below20_Down

def get_electronRECOsf_Above20(events, Sel_Electron, name, period):
    #RECO sf for e
    file_name = {
        '2018': 'UL2018',
        '2017': 'UL2017',
        '2016': 'UL2016postVFP',
        '2016_HIPM': 'UL2016preVFP',
    }
    #RECO sf for e
    fReco_path_ptAbove20 = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/{period}/electron/egammaEffi_ptAbove20-txt_EGM2D_{file_name[period]}_witherr.root' 
    evaluator_Above20 = get_scales_fromjson(fReco_path_ptAbove20)
    mask_Above20 = Sel_Electron.pt >= 20
    e_pt = ak.to_numpy(Sel_Electron.pt[mask_Above20])
    e_pt = np.where(e_pt>500, 500, e_pt)
    RECOsf_Above20 = evaluator_Above20["EGamma_SF2D"](e_pt, ak.to_numpy(Sel_Electron.eta[mask_Above20]))
    RECOsf_Above20_err = evaluator_Above20["EGamma_SF2D_err"](e_pt, ak.to_numpy(Sel_Electron.eta[mask_Above20]))
    RECOsf_Above20_Up = RECOsf_Above20 + RECOsf_Above20_err
    RECOsf_Above20_Down = RECOsf_Above20 - RECOsf_Above20_err
    return  RECOsf_Above20, RECOsf_Above20_Up, RECOsf_Above20_Down
