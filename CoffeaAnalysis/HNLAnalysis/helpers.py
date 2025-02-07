import os
import numpy as np
import csv
import awkward as ak
import json
import uproot
import pickle
import yaml
from CoffeaAnalysis.HNLAnalysis.correction_helpers import get_correction_central

def import_stitching_weights(sample, period):
    ''' Import stitching weights for DY and WJets.
        Stiching weights can be computed from CoffeaAnalysis/stitching code
    '''
    if sample not in ['DYtoLL','WJetsToLNu']:
        raise ('Incorrect arguments for importing stitching weights: ' + sample + ' should be either DYtoLL or WJetsToLNu')
    
    # open json file in read format
    fs = open(f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/stitching/data/{period}/stitching_weights_2D_{sample}.json', "r")
    
    # read data from the file - data.json
    stitching_weights = json.loads(fs.read())

    # close the file
    fs.close()

    return stitching_weights

def reweight_DY(events, stitching_weights_DY):
    ''' Reweight DY samples according to stitching_weights_DY
        using the same PS partialisation for stitching as in CoffeaAnalysis/stitching/DY
    '''
    #set weight to -1/0/1 for uniformity between samples
    np.asarray(events.genWeight)[events.genWeight < 0] = -1.
    np.asarray(events.genWeight)[events.genWeight > 0] = 1.
    np.asarray(events.genWeight)[events.genWeight == 0] = 0.

    #apply stitching weights (depending on the PS in which the event is)
    for NJets in np.arange(len(list(stitching_weights_DY))):
        np.asarray(events.genWeight)[(events.LHE.Vpt == 0) & (events.LHE.NpNLO == int(NJets))] = events.genWeight[(events.LHE.Vpt == 0) & (events.LHE.NpNLO == int(NJets))]*stitching_weights_DY['NJets='+str(NJets)]['PtZ=0']
        np.asarray(events.genWeight)[(events.LHE.Vpt > 0) & (events.LHE.Vpt < 50) & (events.LHE.NpNLO == int(NJets))] = events.genWeight[(events.LHE.Vpt > 0) & (events.LHE.Vpt < 50) & (events.LHE.NpNLO == int(NJets))]*stitching_weights_DY['NJets='+str(NJets)]['PtZ=0to50']
        np.asarray(events.genWeight)[(events.LHE.Vpt >= 50) & (events.LHE.Vpt < 100) & (events.LHE.NpNLO == int(NJets))] = events.genWeight[(events.LHE.Vpt >= 50) & (events.LHE.Vpt < 100) & (events.LHE.NpNLO == int(NJets))]*stitching_weights_DY['NJets='+str(NJets)]['PtZ=50to100']
        np.asarray(events.genWeight)[(events.LHE.Vpt >= 100) & (events.LHE.Vpt < 250) & (events.LHE.NpNLO == int(NJets))] = events.genWeight[(events.LHE.Vpt >= 100) & (events.LHE.Vpt < 250) & (events.LHE.NpNLO == int(NJets))]*stitching_weights_DY['NJets='+str(NJets)]['PtZ=100to250']
        np.asarray(events.genWeight)[(events.LHE.Vpt >= 250) & (events.LHE.Vpt < 400) & (events.LHE.NpNLO == int(NJets))] = events.genWeight[(events.LHE.Vpt >= 250) & (events.LHE.Vpt < 400) & (events.LHE.NpNLO == int(NJets))]*stitching_weights_DY['NJets='+str(NJets)]['PtZ=250to400']
        np.asarray(events.genWeight)[(events.LHE.Vpt >= 400) & (events.LHE.Vpt < 650) & (events.LHE.NpNLO == int(NJets))] = events.genWeight[(events.LHE.Vpt >= 400) & (events.LHE.Vpt < 650) & (events.LHE.NpNLO == int(NJets))]*stitching_weights_DY['NJets='+str(NJets)]['PtZ=400to650']
        np.asarray(events.genWeight)[(events.LHE.Vpt >= 650) & (events.LHE.NpNLO == int(NJets))] = events.genWeight[(events.LHE.Vpt >= 650) & (events.LHE.NpNLO == int(NJets))]*stitching_weights_DY['NJets='+str(NJets)]['PtZ=650toInf']

    return events

def reweight_WJets(events, stitching_weights_WJets):
    ''' Reweight WJets samples according to stitching_weights_WJets 
        using the same PS partialisation for stitching as in CoffeaAnalysis/stitching/WJets
    '''
    #set weight to -1/0/1 for uniformity between samples
    np.asarray(events.genWeight)[events.genWeight < 0] = -1.
    np.asarray(events.genWeight)[events.genWeight > 0] = 1.
    np.asarray(events.genWeight)[events.genWeight == 0] = 0.

    #apply stitching weights (depending on the PS in which the event is)
    for NJets in np.arange(len(list(stitching_weights_WJets))):
        np.asarray(events.genWeight)[(events.LHE.HT == 0) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT == 0) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=0']
        np.asarray(events.genWeight)[(events.LHE.HT > 0) & (events.LHE.HT < 70) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT > 0) & (events.LHE.HT < 70) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=0to70']
        np.asarray(events.genWeight)[(events.LHE.HT >= 70) & (events.LHE.HT < 100) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT >= 70) & (events.LHE.HT < 100) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=70to100']
        np.asarray(events.genWeight)[(events.LHE.HT >= 100) & (events.LHE.HT < 200) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT >= 100) & (events.LHE.HT < 200) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=100to200']
        np.asarray(events.genWeight)[(events.LHE.HT >= 200) & (events.LHE.HT < 400) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT >= 200) & (events.LHE.HT < 400) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=200to400']
        np.asarray(events.genWeight)[(events.LHE.HT >= 400) & (events.LHE.HT < 600) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT >= 400) & (events.LHE.HT < 600) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=400to600']
        np.asarray(events.genWeight)[(events.LHE.HT >= 600) & (events.LHE.HT < 800) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT >= 600) & (events.LHE.HT < 800) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=600to800']
        np.asarray(events.genWeight)[(events.LHE.HT >= 800) & (events.LHE.HT < 1200) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT >= 800) & (events.LHE.HT < 1200) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=800to1200']
        np.asarray(events.genWeight)[(events.LHE.HT >= 1200) & (events.LHE.HT < 2500) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT >= 1200) & (events.LHE.HT < 2500) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=1200to2500']
        np.asarray(events.genWeight)[(events.LHE.HT >= 2500) & (events.LHE.Njets == int(NJets))] = events.genWeight[(events.LHE.HT >= 2500) & (events.LHE.Njets == int(NJets))]*stitching_weights_WJets['NJets='+str(NJets)]['HT=2500toInf']

    return events

def data_goodrun_lumi(ds, era):
    ''' For a given data sample, extract the good runs (golden lumi json from LUMI POG stored in run{era}_lumi.csv file) 
        and the corresponding luminosity stored in run_Data_{era}/ folder

        return a 2D array with the good runs and their corresponding luminosity
    '''
    with open(f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/luminosity/data/goldenrunAndLumi/Run2_{era}_lumi.csv', newline='') as csvfile:
        csv_reader = csv.reader(filter(lambda row: row[0]!='#', csvfile))
        run_goodrun = list(csv_reader)

    #store only information that we need: run number and luminosity
    RunEra_run_lumi = []
    for i in range(len(run_goodrun)):
        RunEra_run_lumi.append([run_goodrun[i][0][0:6],run_goodrun[i][5]])
    RunEra_run_lumi = np.array(RunEra_run_lumi).astype(float)

    #then found the run in the data file (stored in run_Data_ds.csv file)
    run_data = []
    with open(f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/luminosity/data/run_Data/'+era+'/run_'+ds+'.csv', newline='') as csvfile:
        csv_reader = csv.reader(filter(lambda row: row[0]!='#', csvfile))
        run_data.append(list(csv_reader))
    run_data = np.concatenate(run_data).astype(float)

    # do the matching with the "good run" in run{era}_lumi.csv 
    run_lumi = []
    for i in range(len(run_data)):
        result = np.where(RunEra_run_lumi[:,0] == run_data[i])
        if len(result[0]) == 1:
            index = result[0][0]
            run_lumi.append([run_data[i],RunEra_run_lumi[index][1]])
        if len(result[0]) > 1:
            print("WARNING: "+str(ds[:-1])+", run: "+str(run_data[i])+" has multiple matching in run"+era+"_lumi.csv")
    run_lumi = np.array(run_lumi, dtype=object).astype(float)

    return run_lumi

def compute_lumi(ds, period):
    ''' compute exact luminosity in pb for a given data sample area (ex: SingleMuon_{period}A)
    '''
    run_lumi = data_goodrun_lumi(ds, period)
    return np.sum(run_lumi[:,1])*1000 #convertion /fb --> /pb

def compute_reweight(HLT, xsec, event_nb, period):
    ''' compute global weight for the sample (lumi x xsec / n_events)
    ''' 
    all_area_period = {
        '2018': ['A','B','C','D'],
        '2017': ['B','C','D','E','F'],
        '2016': ['F','G','H'],
        '2016_HIPM': ['B','C','D','E','F'],
    }
    #compute luminosity for all area of the given HLT ds used
    luminosity = 0
    for area in all_area_period[period]:
        Data_sample = f'{HLT}_{period}{area}'
        luminosity += compute_lumi(Data_sample, period)
    #print(f'computed luminosity: {luminosity}')

    return luminosity*xsec/event_nb

def apply_MET_Filter(events, period):
    ''' MET filter folowing https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#2018_2017_data_and_MC_UL
    '''
    if period in ['2018','2017']:
        
        events = events[events.Flag.goodVertices & 
                        events.Flag.globalSuperTightHalo2016Filter & 
                        events.Flag.HBHENoiseFilter & 
                        events.Flag.HBHENoiseIsoFilter & 
                        events.Flag.EcalDeadCellTriggerPrimitiveFilter & 
                        events.Flag.BadPFMuonFilter & 
                        events.Flag.BadPFMuonDzFilter & 
                        events.Flag.eeBadScFilter & 
                        events.Flag.ecalBadCalibFilter]
    if period in ['2016','2016_HIPM']:
        events = events[events.Flag.goodVertices & 
                        events.Flag.globalSuperTightHalo2016Filter & 
                        events.Flag.HBHENoiseFilter & 
                        events.Flag.HBHENoiseIsoFilter & 
                        events.Flag.EcalDeadCellTriggerPrimitiveFilter & 
                        events.Flag.BadPFMuonFilter & 
                        #events.Flag.BadPFMuonDzFilter & #not available 
                        events.Flag.eeBadScFilter]
    return events

def apply_reweight(ds, events, stitched_list, HLT, xsecs, period):
    ''' global weight for the sample (lumi x xsec / n_events) + apply stitching weights in case ds is DY or WJets samples
    '''
    #Analysis task can be submited only if counter.pkl exist, which store initial nb of event in ds
    with open (f'{os.getenv("CENTRAL_STORAGE_ANATUPLE")}/anatuple/{period}/counter.pkl', 'rb') as f:
        event_counter = pickle.load(f)

    event_nb = event_counter['sumw_PUcorr'][ds]

    xsec_file = os.path.join(os.getenv("ANALYSIS_PATH"), 'config', 'crossSections13TeV.yaml')
    with open(xsec_file, 'r') as f:
        xsec_load = yaml.safe_load(f)
        xsec_DY_incl = xsec_load["DY_NNLO"]["crossSec"]
        xsec_WJetsToLNu_incl = xsec_load["WJetsToLNu"]["crossSec"]

    #applying stitching weights for DY and WJets samples
    if ds in stitched_list['DY_samples']:
        stitching_weights_DY = import_stitching_weights('DYtoLL', period)
        events = reweight_DY(events, stitching_weights_DY)
        # inclusive xsec and event event_nb for normalisation
        xsecs = xsec_DY_incl
        event_nb  = event_counter['sumw_PUcorr']['DYJetsToLL_M-50']
        
    if ds in stitched_list['WJets_samples']:
        stitching_weights_WJets = import_stitching_weights('WJetsToLNu', period)
        events = reweight_WJets(events, stitching_weights_WJets)
        # inclusive xsec and event event_nb for normalisation
        xsecs = xsec_WJetsToLNu_incl
        event_nb = event_counter['sumw_PUcorr']['WJetsToLNu']

    #reweights events with lumi x xsec / n_events
    scale = compute_reweight(HLT, xsecs, event_nb, period)
    events['genWeight'] = events.genWeight*scale

    return events

def apply_golden_run(ds, events, era):
    ''' For data sample, keep only the good runs ("golden" lumi json from LUMI POG stored in run{era}_lumi.csv file) 
    '''
    goodrun_lumi = data_goodrun_lumi(ds, era)
    goodrun = goodrun_lumi[:,0]
    events = events[np.isin(events.run, goodrun)]
    events['genWeight'] = events.run > 0
    return events

def delta_r2(v1, v2):
    '''Calculates deltaR squared between two particles v1, v2 whose
    eta and phi methods return arrays
    '''
    dphi = (v1.phi - v2.phi + np.pi) % (2 * np.pi) - np.pi
    deta = v1.eta - v2.eta
    dr2 = dphi**2 + deta**2
    return dr2

def delta_r(v1, v2):
    '''Calculates deltaR between two particles v1, v2 whose
    eta and phi methods return arrays.
    
    Note: Prefer delta_r2 for cuts.
    '''
    return np.sqrt(delta_r2(v1, v2))

def bjet_candidates(events, Lepton1, Lepton2, Lepton3, period, delta_r_cut = 0.5):
    ''' Return bjet (btagDeepFlavB > LooseWP) with pt > 20, |eta| < 2.5, jetId >= 2 i.e pass tight or tightLepVeto ID +  do no not match selected lepton
    '''
    # btag wp
    cset = get_correction_central('btag', period)
    #loose WP value
    deepJet_wp_value = cset["deepJet_wp_values"].evaluate("L")
    
    jets_candidates = events.SelJet[(events.SelJet.pt > 20.) & (events.SelJet.eta < 2.5) & (events.SelJet.eta > -2.5) & (events.SelJet.jetId >= 2)]
    drcut_jetslep1 = delta_r(Lepton1,jets_candidates) > delta_r_cut
    drcut_jetslep2 = delta_r(Lepton2,jets_candidates) > delta_r_cut
    drcut_jetslep3 = delta_r(Lepton3,jets_candidates) > delta_r_cut

    # save also info of nbjets when we remove the dR condition with Taus 
    if ('rawDeepTau2017v2p1VSe' in Lepton2.fields) & ('rawDeepTau2017v2p1VSe' in Lepton3.fields):
        bjets_candidates_withoutdRTau = jets_candidates[drcut_jetslep1]
    if ('rawDeepTau2017v2p1VSe' not in Lepton2.fields)  & ('rawDeepTau2017v2p1VSe' in Lepton3.fields) :
        bjets_candidates_withoutdRTau = jets_candidates[drcut_jetslep1 & drcut_jetslep2]
    
    bjets_candidates = jets_candidates[drcut_jetslep1 & drcut_jetslep2 & drcut_jetslep3]

    #save bjets in events
    events['nbjetsLoose'] = ak.num(bjets_candidates[bjets_candidates.btagDeepFlavB > deepJet_wp_value])
    events['nbjetsLooseWithoutdRTau'] = ak.num(bjets_candidates_withoutdRTau[bjets_candidates_withoutdRTau.btagDeepFlavB > deepJet_wp_value])
    events['bjets'] = bjets_candidates

    return 

def bjet_info(events):
    ''' Create a list of all bjets information in events (jagged array managment with uproot)
    '''
    bjet_compress = ak.zip({
        "pt": ak.Array(ak.to_list(events.bjets['pt'])),
        "eta": ak.Array(ak.to_list(events.bjets['eta'])),
        "phi": ak.Array(ak.to_list(events.bjets['phi'])),
        "hadronFlavour": ak.Array(ak.to_list(events.bjets['hadronFlavour'])),
        "btagDeepFlavB": ak.Array(ak.to_list(events.bjets['btagDeepFlavB'])),
        "mass": ak.Array(ak.to_list(events.bjets['mass'])),
        "vetomap": ak.Array(ak.to_list(events.bjets['vetomap']))
    })
        
    return bjet_compress

def save_bjets(save_file, events):
    ''' Save in a root file bjets info in "bjets" Tree
    '''
    if len(events) == 0:
        return
    
    # Split the original string based on the delimiter
    parts = save_file.split("_anatuple_")
    parts[1] = parts[1][:-5]

    while os.path.isfile(save_file):
        save_file = parts[0] + "_anatuple_" + str(int(parts[1])+1) +'.root'

    print(f'... saving bjets in in file {save_file}')
    with uproot.create(save_file, compression=uproot.ZLIB(4)) as file:
        if np.sum(ak.num(bjet_info(events))) != 0:
            file["bjets"] = bjet_info(events)
    return

def save_Event(save_file, lst, Tree_name):
    ''' Save in a root file a Tree (Tree_name) containing lst
    '''
    print(f'... saving events file {save_file}')
    with uproot.create(save_file, compression=uproot.ZLIB(4)) as file:
        file[Tree_name] = lst
    return

def save_anatuple_common(ds, events, tag, period, channel, save_weightcorr):
    ''' Return a list of common information to save (make sure save_file is not overwrite)
    '''
    path = f'/afs/cern.ch/work/p/pdebryas/HNL/tmp/{period}/{tag}/{channel}/{ds}/'

    if not os.path.exists(path):
        os.makedirs(path)

    save_file = path + f'{ds}_anatuple_0.root'

    i = 0
    while os.path.isfile(save_file):
        i = i+1
        save_file = path + f'{ds}_anatuple_{str(i)}.root'

    lst = { "event": np.array(events.event),
            "genWeight": np.array(events.genWeight),
            "luminosityBlock": np.array(events.luminosityBlock),
            "run": np.array(events.run),
            "MET_pt": np.array(events.SelMET['pt']),
            "MET_phi": np.array(events.SelMET['phi']),
            "nbjetsLoose": np.array(events.nbjetsLoose),
            "nbjetsLooseWithoutdRTau": np.array(events.nbjetsLooseWithoutdRTau)
        }

    if save_weightcorr == True:
        weightcorr_list = [element for element in events.fields if element.startswith('weightcorr_')]
        for elem in weightcorr_list:
            lst[elem] = np.array(events[elem])

    return save_file, lst

def save_anatuple_lepton(Sel_Lep, lst, exclude_list, name):
    ''' Update the list with all the informations contains in Sel_Lep object (not in exclude_list)
        name: name of the branch in the rootfile
    '''
    #also add cone corrected p_t
    Sel_Lep[f"ConeCorrectedPt"] = Sel_Lep.pt*(1+Sel_Lep.pfRelIso03_all)
    for field in Sel_Lep.fields:
        if field not in exclude_list:
            lst[f"{name}_{field}"] = np.array(Sel_Lep[field])
        
    return lst

def matching_jet(events, Tau):
    ''' Return matching Jet of Tau object
    Jet valid is True if it exist a Jet that match the tau
    '''
    jetIdx = ak.to_numpy(ak.unflatten(Tau.jetIdx, 1)).data
    jetIdx[jetIdx < 0] = 0
    Jet = events.SelJet[ak.local_index(events.SelJet) == jetIdx]

    Jet['valid'] = Tau.jetIdx >= 0
    return Jet

def matching_Genjet(events, Tau):
    ''' Return the best matching Genjet for Tau object which minimize  Delta r(Tau, GenJet)
        Sel_GenJet valid valid is True if Delta r(Tau, GenJet) < 0.4
    '''
    Sel_GenJet = events.GenJet[delta_r(Tau,events.GenJet) == ak.min(delta_r(Tau,events.GenJet), axis=1)]
    Sel_GenJet['valid'] = ak.min(delta_r(Tau,events.GenJet), axis=1) < 0.4
    
    return Sel_GenJet

def save_anatuple_tau(events, Sel_Tau, lst, exclude_list, mode, name):
    ''' Update the list with all the informations contains in Sel_Tau object (not in exclude_list)
        + matching Jet and GenJet information 
        name: name of the branch in the rootfile
    '''
    if len(Sel_Tau) == 0:
        return lst
    
    if mode != 'Data':
        GenTauMatch = events.GenPart[ak.Array([[i] for i in Sel_Tau['genPartIdx']])]
        Sel_Tau['isPrompt'] = np.array(ak.flatten((GenTauMatch.statusFlags & (1 << 0)) != 0)) #isPrompt
        Sel_Tau['isDirectPromptTauDecayProduct'] = np.array(ak.flatten((GenTauMatch.statusFlags & (1 << 5)) != 0)) #isDirectPromptTauDecayProduct
    #save Tau info
    Sel_Tau['pfRelIso03_all'] = Sel_Tau.rawIsodR03
    lst = save_anatuple_lepton(Sel_Tau, lst, exclude_list, name)

    #save matching Jet info
    if len(Sel_Tau) == 0:
        print('0 events pass selection')
    else:
        Tau_Jet = matching_jet(events, Sel_Tau)
        lst[f"{name}_Jet_pt"] = np.array(ak.flatten(Tau_Jet['pt']))
        lst[f"{name}_Jet_eta"] = np.array(ak.flatten(Tau_Jet['eta']))
        lst[f"{name}_Jet_phi"] = np.array(ak.flatten(Tau_Jet['phi']))
        lst[f"{name}_Jet_mass"] = np.array(ak.flatten(Tau_Jet['mass']))
        lst[f"{name}_Jet_jetId"] = np.array(ak.flatten(Tau_Jet['jetId']))
        lst[f"{name}_Jet_valid"] = np.array(ak.flatten(Tau_Jet['valid']))
    
        if mode != 'Data':
            Tau_GenJet = matching_Genjet(events, Sel_Tau)
            for field in Tau_GenJet.fields:
                lst[f"{name}_GenJet_{field}"] = np.array(Tau_GenJet[field])
        
    return lst

def IsoMuon_mask(events, n, iso_cut = 0.15):
    ''' Return mask to veto events with more than n muon with pfRelIso03_all <= iso_cut
    '''
    Iso_muon = events.SelMuon[events.SelMuon.pfRelIso03_all <= iso_cut]
    return ak.num(Iso_muon) <= n

def IsoElectron_mask(events, n, iso_cut = 0.15):
    ''' Return mask to veto events with more than one electron with pfRelIso03_all <= iso_cut
    '''
    Iso_electron = events.SelElectron[events.SelElectron.pfRelIso03_all <= iso_cut]
    return ak.num(Iso_electron) <= n

def FinalTau_sel(events, Sel_Lepton1, Sel_Lepton2, DeepTauVersion, delta_r_cut = 0.5):
    ''' Select Tau candidate with dr(Sel_Lepton1,Tau)>0.5 and dr(Sel_Lepton2,Tau)>0.5 
        We take the one with maximum rawDeepTauVSjet in case of ambiguity
    '''
    Tau_candidate = events.SelTau[(delta_r(Sel_Lepton1,events.SelTau) > delta_r_cut) & (delta_r(Sel_Lepton2,events.SelTau)> delta_r_cut)]

    #make sure at least 1 satisfy this condition
    cut = ak.num(Tau_candidate) >= 1
    events = events[cut]
    Sel_Lepton1 = Sel_Lepton1[cut]
    Sel_Lepton2 = Sel_Lepton2[cut]
    Tau_candidate = Tau_candidate[cut]

    #Sel_Tau is the one with the highest isolation VSJet, we take the first one by default if same rawDeepTauVSjet
    if DeepTauVersion == 'DeepTau2018v2p5':
        Sel_Tau = Tau_candidate[ak.max(Tau_candidate.rawDeepTau2018v2p5VSjet, axis=-1) == Tau_candidate.rawDeepTau2018v2p5VSjet]
    if DeepTauVersion == 'DeepTau2017v2p1':
        Sel_Tau = Tau_candidate[ak.max(Tau_candidate.rawDeepTau2017v2p1VSjet, axis=-1) == Tau_candidate.rawDeepTau2017v2p1VSjet]
    Sel_Tau = Sel_Tau[:,0]

    return events, Sel_Lepton1, Sel_Lepton2, Sel_Tau

def matching_IsoMu24(events, Sel_Muon, min_dr_cut = 0.2):
    ''' Return Muon among Sel_Muon collection that match Trigger object (IsoMu24 HLT) and within DeltaR(TrigObj,Sel_Muon)<min_dr_cut 
        Trigger object is a muon (id == 13) with pt > 24 (HLT marge) with  TrigObj_filterBits for Muon: 2 = Iso and 8 = 1mu
    '''
    Trigger_Muon = events.TrigObj[ (abs(events.TrigObj.id) == 13)  & (events.TrigObj.pt > 24.) & ((events.TrigObj.filterBits & (2+8)) != 0)]
    trigger_muon, sel_Muon = ak.unzip(ak.cartesian([Trigger_Muon, Sel_Muon], nested=True))

    Sel_Muon = sel_Muon[ak.argmin(delta_r(trigger_muon, sel_Muon), axis=-1, keepdims = True , mask_identity = True)] # Select for each pair (trigger/reco_mu) the reco_mu that match the best with trigger_mu
    Sel_trigger = trigger_muon[ak.argmin(delta_r(trigger_muon, sel_Muon), axis=-1, keepdims = True , mask_identity = True)] # Select for each pair (trigger/reco_mu) the trigger_mu that match the best with reco_mu
    Sel_Muon = Sel_Muon[delta_r(Sel_trigger, Sel_Muon) < min_dr_cut]
    Sel_Muon = ak.flatten(Sel_Muon, axis=-1)

    return Sel_Muon

def matching_IsoMu27(events, Sel_Muon, min_dr_cut = 0.2):
    ''' Return Muon among Sel_Muon collection that match Trigger object (IsoMu27 HLT) and within DeltaR(TrigObj,Sel_Muon)<min_dr_cut 
        Trigger object is a muon (id == 13) with pt > 27 (HLT marge) with  TrigObj_filterBits for Muon: 2 = Iso and 8 = 1mu
    '''
    Trigger_Muon = events.TrigObj[ (abs(events.TrigObj.id) == 13)  & (events.TrigObj.pt > 27.) & ((events.TrigObj.filterBits & (2+8)) != 0)]
    trigger_muon, sel_Muon = ak.unzip(ak.cartesian([Trigger_Muon, Sel_Muon], nested=True))

    Sel_Muon = sel_Muon[ak.argmin(delta_r(trigger_muon, sel_Muon), axis=-1, keepdims = True , mask_identity = True)] # Select for each pair (trigger/reco_mu) the reco_mu that match the best with trigger_mu
    Sel_trigger = trigger_muon[ak.argmin(delta_r(trigger_muon, sel_Muon), axis=-1, keepdims = True , mask_identity = True)] # Select for each pair (trigger/reco_mu) the trigger_mu that match the best with reco_mu
    Sel_Muon = Sel_Muon[delta_r(Sel_trigger, Sel_Muon) < min_dr_cut]
    Sel_Muon = ak.flatten(Sel_Muon, axis=-1)

    return Sel_Muon

def Trigger_Muon_sel(events, period):
    ''' Select Trigger muon for IsoMu24 HLT
        Require min pt/eta for trigger obj matching for at leat 1 muon. 
        According to the central muon SFs recommendations it should be online pt + 2 GeV and |eta| < 2.1
        Select one Muon that match trigger object
        We take the one with minimum pfRelIso03_all in case of ambiguity
    '''
    if period == '2018':
        Trigger_Muon_candidates = events.SelMuon[(events.SelMuon.pt > 26.) & (np.abs(events.SelMuon.eta) < 2.1)]
    if period == '2017':
        Trigger_Muon_candidates = events.SelMuon[(events.SelMuon.pt > 29.) & (np.abs(events.SelMuon.eta) < 2.1)]
    if period == '2016':
        Trigger_Muon_candidates = events.SelMuon[(events.SelMuon.pt > 26.) & (np.abs(events.SelMuon.eta) < 2.1)]
    if period == '2016_HIPM':
        Trigger_Muon_candidates = events.SelMuon[(events.SelMuon.pt > 26.) & (np.abs(events.SelMuon.eta) < 2.1)]

    #make sure at least 1 muon satisfy this condition
    cut_mask = ak.num(Trigger_Muon_candidates) >= 1
    events = events[cut_mask]
    Trigger_Muon_candidates = Trigger_Muon_candidates[cut_mask]

    #select the one that match trigger 
    if period == '2018':
        Trigger_Muon_candidates = matching_IsoMu24(events, Trigger_Muon_candidates)
    if period == '2017':
        Trigger_Muon_candidates = matching_IsoMu27(events, Trigger_Muon_candidates)
    if period == '2016':
        Trigger_Muon_candidates = matching_IsoMu24(events, Trigger_Muon_candidates)
    if period == '2016_HIPM':
        Trigger_Muon_candidates = matching_IsoMu24(events, Trigger_Muon_candidates)

    cut_mask = ak.num(Trigger_Muon_candidates) >= 1
    events = events[cut_mask]
    Trigger_Muon_candidates = Trigger_Muon_candidates[cut_mask]

    #first muons selected with minimum pfRelIso03_all. In case same isolation, we take the first one by default
    Sel_Muon = Trigger_Muon_candidates[ak.min(Trigger_Muon_candidates.pfRelIso03_all, axis=-1) == Trigger_Muon_candidates.pfRelIso03_all]
    Sel_Muon = Sel_Muon[:,0]

    return events, Sel_Muon

def matching_Ele32(events, Sel_Electron, min_dr_cut = 0.2):
    ''' Return Electron among Sel_Electron collection that match Trigger object (Ele32 HLT) and within DeltaR(TrigObj,Sel_Electron)<min_dr_cut
        Trigger object is a electron (id == 11) with TrigObj_filterBits 2= 1e WPTight
    '''
    Trigger_Electron = events.TrigObj[ (abs(events.TrigObj.id) == 11) & ((events.TrigObj.filterBits & (2)) != 0) & (events.TrigObj.pt > 32.)]  
    trigger_electron, sel_electron = ak.unzip(ak.cartesian([Trigger_Electron, Sel_Electron], nested=True))

    Sel_Electron = sel_electron[ak.argmin(delta_r(trigger_electron, sel_electron), axis=-1, keepdims = True , mask_identity = True)] # Select for each pair (trigger/reco_e) the reco_e that match the best with trigger_e
    Sel_trigger = trigger_electron[ak.argmin(delta_r(trigger_electron, sel_electron), axis=-1, keepdims = True , mask_identity = True)] 
    Sel_Electron = Sel_Electron[delta_r(Sel_trigger, Sel_Electron) < min_dr_cut]
    Sel_Electron = ak.flatten(Sel_Electron, axis=-1)
    
    return Sel_Electron

def matching_Ele25(events, Sel_Electron, min_dr_cut = 0.2):
    ''' Return Electron among Sel_Electron collection that match Trigger object (Ele25 HLT) and within DeltaR(TrigObj,Sel_Electron)<min_dr_cut
        Trigger object is a electron (id == 11) with TrigObj_filterBits 2= 1e WPTight
    '''
    Trigger_Electron = events.TrigObj[ (abs(events.TrigObj.id) == 11) & ((events.TrigObj.filterBits & (2)) != 0) & (events.TrigObj.pt > 25.)]   
    trigger_electron, sel_electron = ak.unzip(ak.cartesian([Trigger_Electron, Sel_Electron], nested=True))

    Sel_Electron = sel_electron[ak.argmin(delta_r(trigger_electron, sel_electron), axis=-1, keepdims = True , mask_identity = True)] # Select for each pair (trigger/reco_e) the reco_e that match the best with trigger_e
    Sel_trigger = trigger_electron[ak.argmin(delta_r(trigger_electron, sel_electron), axis=-1, keepdims = True , mask_identity = True)] 
    Sel_Electron = Sel_Electron[delta_r(Sel_trigger, Sel_Electron) < min_dr_cut]
    Sel_Electron = ak.flatten(Sel_Electron, axis=-1)
    
    return Sel_Electron

def Trigger_Electron_sel(events, period):
    ''' Select Trigger electron for Ele32 HLT
        Require min pt/eta for trigger obj matching for at leat 1 electron. 
        According to the central electron SFs recommendations it should be online pt + 2 GeV 
        Select one Electron that match trigger object
        We take the one with minimum pfRelIso03_all in case of ambiguity
    '''
    if period == '2018':
        Trigger_Electron_candidates = events.SelElectron[(events.SelElectron.pt > 34.)]
    if period == '2017':
        Trigger_Electron_candidates = events.SelElectron[(events.SelElectron.pt > 34.)]
    if period == '2016':
        Trigger_Electron_candidates = events.SelElectron[(events.SelElectron.pt > 27.) & (np.abs(events.SelElectron.eta) < 2.1)]
    if period == '2016_HIPM':
        Trigger_Electron_candidates = events.SelElectron[(events.SelElectron.pt > 27.) & (np.abs(events.SelElectron.eta) < 2.1)]

    #make sure at least 1 muon satisfy this condition
    cut_mask = ak.num(Trigger_Electron_candidates) >= 1
    events = events[cut_mask]
    Trigger_Electron_candidates = Trigger_Electron_candidates[cut_mask]

    #select the one that match trigger 
    if period == '2018':
        Trigger_Electron_candidates = matching_Ele32(events, Trigger_Electron_candidates)
    if period == '2017':
        Trigger_Electron_candidates = matching_Ele32(events, Trigger_Electron_candidates)
    if period == '2016':
        Trigger_Electron_candidates = matching_Ele25(events, Trigger_Electron_candidates)
    if period == '2016_HIPM':
        Trigger_Electron_candidates = matching_Ele25(events, Trigger_Electron_candidates)

    cut_mask = ak.num(Trigger_Electron_candidates) >= 1
    events = events[cut_mask]
    Trigger_Electron_candidates = Trigger_Electron_candidates[cut_mask]

    #first electron selected with minimum pfRelIso03_all. In case same isolation, we take the first one by default
    Sel_Electron = Trigger_Electron_candidates[ak.min(Trigger_Electron_candidates.pfRelIso03_all, axis=-1) == Trigger_Electron_candidates.pfRelIso03_all]
    Sel_Electron = Sel_Electron[:,0]

    return events, Sel_Electron

def add_gen_matching_info(events, lepton):
    """
    Adds GenPart matching information to the specified object collection in an Awkward array.

    Parameters:
        events (ak.Array): The Awkward array containing NanoAOD events.
        lepton (ak.Array): The collection of the lepton.

    Returns:
        ak.Array: The updated lepton array with GenPart matching information.
    """
    genparts = events['GenPart']

    # Get the GenPart indices from the object collection
    gen_indices = lepton.genPartIdx

    matched_part = genparts[ak.local_index(genparts, axis=1) == gen_indices]

    # Mask invalid indices (e.g., -1 means no match)
    valid_mask = (gen_indices >= 0) & (gen_indices < ak.num(genparts, axis=1))
    matched_genparts = ak.mask(matched_part, valid_mask)  # Use ak.mask instead of ak.where

    # List of GenPart attributes to keep
    genpart_features = ["pt", "eta", "phi", "mass", "pdgId", "status", "statusFlags"]

    # Add GenPart attributes to the object collection with the naming convention
    for feature in genpart_features:
        feature_data = matched_genparts[feature]
        # Preserve structure and ensure correct length
        lepton[f"MatchingGenPart_{feature}"] = ak.fill_none(ak.firsts(feature_data), np.nan)      
    return lepton
