import os
import numpy as np
import csv
import awkward as ak
import collections
from coffea import processor, hist
import json

# -------------------------------------------------- generaly useful functions --------------------------------------------------
def import_stitching_weights(sample):

    if sample not in ['DYtoLL','WJetsToLNu']:
        raise ('Incorrect arguments for importing stitching weights: '+ sample)
    
    # open json file in read format
    fs = open(f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/stitching/stitching_weights_2D_{sample}.json', "r")
    
    # read data from the file - data.json
    stitching_weights = json.loads(fs.read())

    # close the file
    fs.close()

    return stitching_weights

def data_goodrun_lumi(ds):
    # for a given data sample, extract the good runs and the corresponding luminosity
    # good run for 2018 are stored in run2018_lumi.csv file
    with open(f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/luminosity/run2018_lumi.csv', newline='') as csvfile:
        csv_reader = csv.reader(filter(lambda row: row[0]!='#', csvfile))
        run2018_goodrun = list(csv_reader)

    #store only information that we need: run number and luminosity
    run2018_run_lumi = []
    for i in range(len(run2018_goodrun)):
        run2018_run_lumi.append([run2018_goodrun[i][0][0:6],run2018_goodrun[i][5]])
    run2018_run_lumi = np.array(run2018_run_lumi).astype(float)

    #then found the run in the data file (stored in run_Data_ds.csv file)
    run_data = []
    with open(f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/luminosity/run_Data/run_'+ds+'.csv', newline='') as csvfile:
        csv_reader = csv.reader(filter(lambda row: row[0]!='#', csvfile))
        run_data.append(list(csv_reader))
    run_data = np.concatenate(run_data).astype(float)

    # do the matching with the "good run" in run2018_lumi.csv
    run_lumi = []
    for i in range(len(run_data)):
        result = np.where(run2018_run_lumi[:,0] == run_data[i])
        if len(result[0]) == 1:
            index = result[0][0]
            run_lumi.append([run_data[i],run2018_run_lumi[index][1]])
        #if len(result[0]) == 0:
            #print("no good run found in run2018_lumi.csv for "+str(ds[:-1])+", run: "+str(run_data[i]))
        if len(result[0]) > 1:
            print("WARNING: "+str(ds[:-1])+", run: "+str(run_data[i])+" has multiple matching in run2018_lumi.csv")
    run_lumi = np.array(run_lumi, dtype=object).astype(float)
    #return an array with good runs and their corresponding luminosity
    return run_lumi

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

def delta_phi(v1, v2):
    '''Calculates delta phi  between two particles v1, v2 whose
    phi method return arrays
    '''
    return (v1.phi - v2.phi + np.pi) % (2 * np.pi) - np.pi

def delta_eta(v1, v2):
    '''Calculates delta eta  between two particles v1, v2 whose
    eta method return arrays
    '''
    return (v1.eta - v2.eta + np.pi) % (2 * np.pi) - np.pi

def cos_opening_angle(v, pv, p1, p2):
    '''Calculates cosine of opening angle between passed
    vertex v (with methods x, y, z), primary vertex pv,
    and the four-vector sum of two passed particles
    p1 and p2 (with methods pt/eta/phi)
    '''
    x = v.vtx_x - pv.x
    y = v.vtx_y - pv.y
    z = v.vtx_z - pv.z
    px = (p1.pt * np.cos(p1.phi) + p2.pt * np.cos(p2.phi))
    py = (p1.pt * np.sin(p1.phi) + p2.pt * np.sin(p2.phi))
    pz = (p1.pt * np.sinh(p1.eta) + p2.pt * np.sinh(p2.eta))
    
    num = x*px + y*py + z*pz
    den = np.sqrt(x**2 + y**2 + z**2) * np.sqrt(px**2 + py**2 + pz**2)
    return num/den

def dxy_significance(v, pv):
    dxy2 = (v.vtx_x - pv.x)**2 + (v.vtx_y - pv.y)**2
    edxy2 = v.vtx_ex**2 + v.vtx_ey**2 #PV has no but anyway negligible error
    return np.sqrt(dxy2/edxy2)

def inv_mass(p1, p2):
    '''Calculates invariant mass from 
    two input particles in pt/eta/phi
    representation, assuming zero mass 
    (so it works for track input)'''
    px1 = p1.pt * np.cos(p1.phi)
    px2 = p2.pt * np.cos(p2.phi)
    px = px1 + px2

    py1 = p1.pt * np.sin(p1.phi) 
    py2 = p2.pt * np.sin(p2.phi)
    py = py1+ py2
    pz1 = p1.pt * np.sinh(p1.eta)
    pz2 = p2.pt * np.sinh(p2.eta)
    pz = pz1 + pz2
    #e = np.sqrt(p1.pt**2 + pz1**2) + np.sqrt(p2.pt**2 + pz2**2)
    e = np.sqrt(px1**2 + py1**2 + pz1**2) + np.sqrt(px2**2 + py2**2 + pz2**2)
    e2 = e**2
    m2 = e2 - px**2 - py**2 - pz**2
    
    if(m2 < 0):
        return 0
    else:
        return np.sqrt(m2)

def inv_mass_3p(p1, p2, p3):
    '''Calculates invariant mass from 
    three input particles in pt/eta/phi
    representation, assuming zero mass 
    (so it works for track input)'''
    x = p1.pt * np.cos(p1.phi) + p2.pt * np.cos(p2.phi) + p3.pt * np.cos(p3.phi)
    y = p1.pt * np.sin(p1.phi) + p2.pt * np.sin(p2.phi) + p3.pt * np.sin(p3.phi)
    z = p1.pt * np.sinh(p1.eta) + p2.pt * np.sinh(p2.eta) + p3.pt * np.sinh(p3.eta)
    e = np.sqrt(p1.pt**2 + (p1.pt * np.sinh(p1.eta))**2) + np.sqrt(p2.pt**2 + (p2.pt * np.sinh(p2.eta))**2) + np.sqrt(p3.pt**2 + (p3.pt * np.sinh(p3.eta))**2)
    m2 = e**2 - x**2 - y**2 - z**2
    return np.sqrt(m2)

def files_from_dir(d):
    '''Returns a list of all ROOT files in the passed directory.
    '''
    files = os.listdir(d)
    return ['/'.join([d, f]) for f in files if f.endswith('.root')]

def HNL_from_dir(d, mass):
    '''Returns a list of all ROOT files in the passed directory.
    '''
    files = os.listdir(d)
    return ['/'.join([d, f]) for f in files if f.endswith(str(mass)+'.root')]

def one_file_from_dir(d):
    '''Returns a list of all ROOT files in the passed directory.
    '''
    files = os.listdir(d)
    files = files[:1]
    return ['/'.join([d, f]) for f in files if f.endswith('.root')]

def files_from_dirs(dirs):
    '''Returns a list of all ROOT files from the directories in the passed list.
    '''
    files = []
    for d in dirs:
        files += files_from_dir(d)
    return files

def files_from_path(path):
    '''Returns a list of all ROOT files from a path.
    '''
    if path.endswith('.root'):
        files = []
        files.append(path)
        return files
    else:
        files = os.listdir(path)
        return ['/'.join([path, f]) for f in files if (f.endswith('.root') and not f.startswith('.'))]

# -------------------------------------------------- functions for histograms --------------------------------------------------

def saved_leading_muon(events, Sel_Muon, out, ds):
    #print("save_muon " + ds)
    out[f'pt_mu1'].fill(ds=ds, pt=ak.flatten(Sel_Muon.pt, axis=None), weight=events.genWeight)
    out[f'eta_mu1'].fill(ds=ds, eta=ak.flatten(Sel_Muon.eta, axis=None), weight=events.genWeight)
    out[f'phi_mu1'].fill(ds=ds, phi=ak.flatten(Sel_Muon.phi, axis=None), weight=events.genWeight)
    out[f'mass_mu1'].fill(ds=ds, mass=ak.flatten(Sel_Muon.mass, axis=None), weight=events.genWeight)

def saved_subleading_muon(events, Sel_Muon, out, ds):
    #print("save_muon " + ds)
    out[f'pt_mu2'].fill(ds=ds, pt=ak.flatten(Sel_Muon.pt, axis=None), weight=events.genWeight)
    out[f'eta_mu2'].fill(ds=ds, eta=ak.flatten(Sel_Muon.eta, axis=None), weight=events.genWeight)
    out[f'phi_mu2'].fill(ds=ds, phi=ak.flatten(Sel_Muon.phi, axis=None), weight=events.genWeight)
    out[f'mass_mu2'].fill(ds=ds, mass=ak.flatten(Sel_Muon.mass, axis=None), weight=events.genWeight)

def saved_leading_electron(events, Sel_Electron, out, ds):
    #print("save_electron " + ds)
    out[f'pt_e1'].fill(ds=ds, pt=ak.flatten(Sel_Electron.pt, axis=None), weight=events.genWeight)
    out[f'eta_e1'].fill(ds=ds, eta=ak.flatten(Sel_Electron.eta, axis=None), weight=events.genWeight)
    out[f'phi_e1'].fill(ds=ds, phi=ak.flatten(Sel_Electron.phi, axis=None), weight=events.genWeight)
    out[f'mass_e1'].fill(ds=ds, mass=ak.flatten(Sel_Electron.mass, axis=None), weight=events.genWeight)

def saved_subleading_electron(events, Sel_Electron, out, ds):
    #print("save_electron " + ds)
    out[f'pt_e2'].fill(ds=ds, pt=ak.flatten(Sel_Electron.pt, axis=None), weight=events.genWeight)
    out[f'eta_e2'].fill(ds=ds, eta=ak.flatten(Sel_Electron.eta, axis=None), weight=events.genWeight)
    out[f'phi_e2'].fill(ds=ds, phi=ak.flatten(Sel_Electron.phi, axis=None), weight=events.genWeight)
    out[f'mass_e2'].fill(ds=ds, mass=ak.flatten(Sel_Electron.mass, axis=None), weight=events.genWeight)

def saved_leading_tau(events, Sel_Tau, out, ds):
    #print("save_tau " + ds)
    out[f'pt_tau1'].fill(ds=ds, pt=ak.flatten(Sel_Tau.pt, axis=None), weight=events.genWeight)
    out[f'eta_tau1'].fill(ds=ds, eta=ak.flatten(Sel_Tau.eta, axis=None), weight=events.genWeight)
    out[f'phi_tau1'].fill(ds=ds, phi=ak.flatten(Sel_Tau.phi, axis=None), weight=events.genWeight)
    out[f'mass_tau1'].fill(ds=ds, mass=ak.flatten(Sel_Tau.mass, axis=None), weight=events.genWeight)

def saved_subleading_tau(events, Sel_Tau, out, ds):
    #print("save_tau " + ds)
    out[f'pt_tau2'].fill(ds=ds, pt=ak.flatten(Sel_Tau.pt, axis=None), weight=events.genWeight)
    out[f'eta_tau2'].fill(ds=ds, eta=ak.flatten(Sel_Tau.eta, axis=None), weight=events.genWeight)
    out[f'phi_tau2'].fill(ds=ds, phi=ak.flatten(Sel_Tau.phi, axis=None), weight=events.genWeight)
    out[f'mass_tau2'].fill(ds=ds, mass=ak.flatten(Sel_Tau.mass, axis=None), weight=events.genWeight)

def saved_subsubleading_tau(events, Sel_Tau, out, ds):
    #print("save_tau " + ds)
    out[f'pt_tau3'].fill(ds=ds, pt=ak.flatten(Sel_Tau.pt, axis=None), weight=events.genWeight)
    out[f'eta_tau3'].fill(ds=ds, eta=ak.flatten(Sel_Tau.eta, axis=None), weight=events.genWeight)
    out[f'phi_tau3'].fill(ds=ds, phi=ak.flatten(Sel_Tau.phi, axis=None), weight=events.genWeight)
    out[f'mass_tau3'].fill(ds=ds, mass=ak.flatten(Sel_Tau.mass, axis=None), weight=events.genWeight)

def saved_dilepton_mass(events, Lepton1, Lepton2, out, ds):
    #print("dilepton_mass " + ds)
    out[f'comb_mass_l1l2'].fill(ds=ds, mc=ak.flatten((Lepton1 + Lepton2).mass, axis=None), weight=events.genWeight)

def saved_dilepton_mass_taul1_OS(events, Lepton, Tau1, Tau2, out, ds):
    # +++ and --- events not recorded!

    #Lepton and Tau1 OS (and tau2 SS as Lepton)
    sel1 = ak.flatten((Tau1.charge != Tau2.charge) & (Lepton.charge != Tau1.charge))
    out[f'comb_mass_taul1'].fill(ds=ds, mc=ak.flatten((Lepton[sel1] + Tau1[sel1]).mass, axis=None), weight=events[sel1].genWeight)
    #Lepton and Tau2 OS (and tau1 SS as Lepton)
    sel2 = ak.flatten((Tau1.charge != Tau2.charge) & (Lepton.charge != Tau2.charge))
    out[f'comb_mass_taul1'].fill(ds=ds, mc=ak.flatten((Lepton[sel2] + Tau2[sel2]).mass, axis=None), weight=events[sel2].genWeight)
    #Lepton and Tau2/Tau1 OS --> take the leading tau
    sel3 = ak.flatten((Tau1.charge == Tau2.charge) & (Lepton.charge != Tau2.charge) & (Tau1.pt >= Tau2.pt))
    out[f'comb_mass_taul1'].fill(ds=ds, mc=ak.flatten((Lepton[sel3] + Tau1[sel3]).mass, axis=None), weight=events[sel3].genWeight)
    sel4 = ak.flatten((Tau1.charge == Tau2.charge) & (Lepton.charge != Tau2.charge) & (Tau1.pt < Tau2.pt))
    out[f'comb_mass_taul1'].fill(ds=ds, mc=ak.flatten((Lepton[sel4] + Tau2[sel4]).mass, axis=None), weight=events[sel4].genWeight)

    #check orthogonality/sanity check
    sel5 = ak.flatten((Tau1.charge == Tau2.charge) & (Lepton.charge == Tau2.charge))
    if (len(events) - (ak.sum(sel1)+ak.sum(sel2)+ak.sum(sel3)+ak.sum(sel4)+ak.sum(sel5))) != 0 :
        print('problem with saved_dilepton_mass_taul1_OS for ds:' + ds)

def saved_drl1l2(events, Lepton1, Lepton2, out, ds):
    #print("save_drl1l2 " + ds)
    out[f'dr_l1l2'].fill(ds=ds, dr=ak.flatten(delta_r(Lepton1,Lepton2), axis=None), weight=events.genWeight)

def saved_MET(events, out, ds):
    #print("save_drl1l2 " + ds)
    out[f'met'].fill(ds=ds, met=events.MET.pt, weight=events.genWeight)

def saved_pt_sum_l1l2l3(events, Lepton1, Lepton2, Lepton3, out, ds):
    out[f'pt_sum_l1l2l3'].fill(ds=ds, pt=ak.flatten((Lepton1.pt + Lepton2.pt  + Lepton3.pt), axis=None), weight=events.genWeight)

def saved_pt_sum_l1l2MET(events, Lepton1, Lepton2, out, ds):
    out[f'pt_sum_l1l2MET'].fill(ds=ds, pt=ak.flatten((Lepton1.pt + Lepton2.pt  + events.MET.pt), axis=None), weight=events.genWeight)

def saved_mT_tautau(events, Lepton1, Lepton2, out, ds):
    mT_tautau = np.sqrt( Lepton1.mass**2 + Lepton2.mass**2 + 2*(np.sqrt(Lepton1.mass**2 + Lepton1.pt**2)*np.sqrt(Lepton2.mass**2 + Lepton2.pt**2) - Lepton1.pt*Lepton2.pt*np.cos(abs(Lepton1.phi - Lepton2.phi))))
    out[f'mT_tautau'].fill(ds=ds, mt=ak.flatten(mT_tautau, axis=None), weight=events.genWeight)

def saved_mT_l1MET(events, Lepton1, MET, out, ds):
    mT_l1MET = np.sqrt( Lepton1.mass**2 + 2*(np.sqrt(Lepton1.mass**2 + Lepton1.pt**2)*MET.pt - Lepton1.pt*MET.pt*np.cos(abs(Lepton1.phi - MET.phi))))
    out[f'mT_l1MET'].fill(ds=ds, mt=ak.flatten(mT_l1MET, axis=None), weight=events.genWeight)

# -------------------------------------------------- functions for lepton selection --------------------------------------------------

def select_lep1_IsoMu24(events, min_dr_cut = 0.2):
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

def select_lep1_Ele32_WPTight_Gsf_L1DoubleEG(events, min_dr_cut = 0.2):
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

def select_lep1_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_old(events, min_dr_cut=0.2):

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

def select_lep_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg(events, min_dr_cut=0.2, min_pt_cut=40.):
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

    # impose min pt > 40 GeV for Tau1 and Tau 2
    cut_pt_mask = ak.any((Sel_Tau1.pt > min_pt_cut) & (Sel_Tau2.pt > min_pt_cut), axis=1)
    Sel_Tau1 = Sel_Tau1[cut_pt_mask]
    Sel_Tau2 = Sel_Tau2[cut_pt_mask]
    events = events[cut_pt_mask]
    
    return events, Sel_Tau1, Sel_Tau2

def select_lep2(events, Sel_Lep1, type, delta_r_cut = 0.5):

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

def select_lep3(events, Sel_Lep1, Sel_Lep2, type, delta_r_cut = 0.5):

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

def bjet_veto(events, Lepton1, Lepton2, Lepton3, delta_r_cut = 0.5):
    # Reject events in order to reduce the background from processes with top quarks (producing jets)
    # the dr non-matching with lepton are here to make sure jet is not misidentify as a signal lepton
    bjets_candidates = events.Jet[(events.Jet.pt > 20.) & (events.Jet.eta < 2.5) & (events.Jet.jetId >= 4) & (events.Jet.btagDeepFlavB > 0.0490)]

    sel_lep1, sel_jets = ak.unzip(ak.cartesian([Lepton1, bjets_candidates], nested=True))
    drcut_jetslep1 = (delta_r(sel_lep1,sel_jets) > delta_r_cut)

    sel_lep2, sel_jets = ak.unzip(ak.cartesian([Lepton2, bjets_candidates], nested=True))
    drcut_jetslep2 = (delta_r(sel_lep2,sel_jets) > delta_r_cut)

    sel_lep3, sel_jets = ak.unzip(ak.cartesian([Lepton3, bjets_candidates], nested=True))
    drcut_jetslep3 = (delta_r(sel_lep3,sel_jets) > delta_r_cut)

    nonmatching_lep_jets = sel_jets[drcut_jetslep1 & drcut_jetslep2 & drcut_jetslep3]
    nonmatching_lep_jets = ak.fill_none(nonmatching_lep_jets[:,0], [], axis=0)

    cut = (ak.num(nonmatching_lep_jets) == 0)
    events = events[cut]
    Lepton1 = Lepton1[cut]
    Lepton2 = Lepton2[cut]
    Lepton3 = Lepton3[cut]
    return events, Lepton1, Lepton2, Lepton3

def charge_veto(events, Lepton1, Lepton2, Lepton3):
    # remove +++ and --- events
    cut = ak.flatten( (Lepton1.charge != Lepton2.charge) | (Lepton1.charge != Lepton3.charge))
    events = events[cut]
    Lepton1 = Lepton1[cut]
    Lepton2 = Lepton2[cut]
    Lepton3 = Lepton3[cut]
    return events, Lepton1, Lepton2, Lepton3

def met_veto(events, Lepton1, Lepton2, Lepton3):
    metcut = 25.
    cut = events.MET.pt > metcut
    events = events[cut]
    Lepton1 = Lepton1[cut]
    Lepton2 = Lepton2[cut]
    Lepton3 = Lepton3[cut]
    return events, Lepton1, Lepton2, Lepton3

def z_veto_tll(events, Tau1, Lepton1, Lepton2):
    # work for tee tmm and tem_OS channel (for tem_SS just invert Tau1 with l1 or l2 as we removed +++ --- events)
    mass_z = 91.2 #GeV
    interval = 10.
    cut = ak.flatten( ((Lepton1 + Lepton2).mass < (mass_z - interval)) |  ((Lepton1 + Lepton2).mass > (mass_z + interval)) )
    events = events[cut]
    Tau1 = Tau1[cut]
    Lepton1 = Lepton1[cut]
    Lepton2 = Lepton2[cut]
    return events, Tau1, Lepton1, Lepton2

def z_veto_ttl(events, Tau1, Tau2, Lepton):
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

def reweight_DY(events, stitching_weights_DY):
    # reweight DY samples according to stitching_weights_DY        
    np.asarray(events.genWeight)[events.genWeight < 0] = -1.
    np.asarray(events.genWeight)[events.genWeight > 0] = 1.
    np.asarray(events.genWeight)[events.genWeight == 0] = 0.

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
    # reweight WJets samples according to stitching_weights_WJets
    np.asarray(events.genWeight)[events.genWeight < 0] = -1.
    np.asarray(events.genWeight)[events.genWeight > 0] = 1.
    np.asarray(events.genWeight)[events.genWeight == 0] = 0.

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