import os
import numpy as np
import csv

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
        return ['/'.join([path, f]) for f in files if f.endswith('.root')]

def data_goodrun_lumi(ds):
    '''for a given data sample, extract the run numbers of the NanoAOD samples
    then see if there is a match with the run for 2018 (stored in run2018_lumi.csv).
    return run number with a match and their corresponding luminosity
    '''
    # open the file which have run number <-> luminosity correspondancy and store only information that we need: run number and luminosity
    with open('luminosity/run2018_lumi.csv', newline='') as csvfile:
        csv_reader = csv.reader(filter(lambda row: row[0]!='#', csvfile))
        run2018_lumi = list(csv_reader)
    run2018_run_lumi = []
    for i in range(len(run2018_lumi)):
        run2018_run_lumi.append([run2018_lumi[i][0][0:6],run2018_lumi[i][5]])
    run2018_run_lumi = np.array(run2018_run_lumi).astype(float)

    #then found the run numbers of the NanoAOD samples (stored in run_Data_ds.csv file)
    run_data = []
    with open('luminosity/run_Data/run_'+ds+'.csv', newline='') as csvfile:
        csv_reader = csv.reader(filter(lambda row: row[0]!='#', csvfile))
        run_data.append(list(csv_reader))
    run_data = np.concatenate(run_data).astype(float)

    # do the matching with the run in run2018_lumi.csv
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
    run_lumi = np.array(run_lumi, dtype='float').astype(float)
    #return an array with good runs and their corresponding luminosity
    return run_lumi