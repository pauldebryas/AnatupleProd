import hist
import os
import correctionlib.convert
import numpy as np
#import awkward as ak
import matplotlib.pyplot as plt
import uproot

def load_ntuples(samples):
  processes = samples.keys()
  branches = {}
  for p in processes:
      DataUproot = uproot.open(samples[p])
      if 'bjets;1' in DataUproot.keys():
        branches[p] = DataUproot['bjets;1'].arrays()
      else:
         print(f'no bjets found in {p}')
         
  return branches

def get_correction_central(period):
    Area_dir = {
        '2016_HIPM': '2016preVFP_UL',
        '2016': '2016postVFP_UL',
        '2017': '2017_UL',
        '2018': '2018_UL'
    }
    POG_path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/"
    #load correction from central repo
    f_path = os.path.join(POG_path, 'BTV/' + Area_dir[period] + '/btagging.json.gz')
    ceval = correctionlib.CorrectionSet.from_file(f_path)
    return ceval

def equalObs(x, nbin):
    #calculate equal-frequency bins 
    nlen = len(x)
    return np.array(np.interp(np.linspace(0, nlen, nbin + 1), np.arange(nlen), np.sort(x)))

def make_hist_sf(den_pt, den_eta, num_pt, num_eta, n_bins, output_figures):
  #pt_bins = np.array( equalObs( den_pt, n_bins) )
  # if pt_bins[-2] < 1000:
  #    pt_bins[-1] = 1000
  # print(pt_bins)
  #eta_bins = np.array( equalObs( np.abs(den_eta) , n_bins) )
  pt_bins = np.array( [20.,25., 30., 45.,75., 1000.] )
  eta_bins = np.array( [0,0.5,1.0,1.5,2.0,2.5] )
  dists = (
      hist.Hist.new
      .StrCat(["num", "den"], name="selection", growth=True)
      .Variable(pt_bins, name="pt")
      .Variable(eta_bins, name="abs_eta")
      .Weight()
      .fill(
          selection="num",
          pt=num_pt,
          abs_eta=np.abs(num_eta)
      )
      .fill(
          selection="den",
          pt=den_pt,
          abs_eta=np.abs(den_eta)
      )
  )
  num = np.array(dists["num", :, :].values())
  den = np.array(dists["den", :, :].values())
  err_num = np.where( num > 0, np.sqrt(num), 0)
  err_den =  np.where( den > 0, np.sqrt(den), 0)

  #We will set it to 0 anywhere we run out of statistics for the correction, to avoid divide by zero issues.
  sf = np.where((den > 0), num / den, 0)
  err_sf = np.where((num > 0) & (den > 0), sf*np.sqrt( (err_num/num)**2 + (err_den/den)**2 ), 0) 

  sfhist = hist.Hist(*dists.axes[1:], data=sf)
  err_sfhist = hist.Hist(*dists.axes[1:], data=err_sf)

  # without a name, the resulting object will fail validation
  sfhist.name = "BTag_eff"
  err_sfhist.name = 'BTag_eff_err'
  sfhist.label = "eff"
  err_sfhist.label = 'eff_err'

  #plotting fake rate
  fig, ax = plt.subplots(figsize=(10, 10))
  sfhist.plot2d(ax=ax)

  for i in range(len(eta_bins)-1):
      for j in range(len(pt_bins)-1):
          str = f"{round(sfhist.values().T[i,j],2)} \n $\pm$ {round(err_sfhist.values().T[i,j],2)}" 
          ax.text(pt_bins[j]+((pt_bins[j+1]-pt_bins[j])/2),eta_bins[i]+((eta_bins[i+1]-eta_bins[i])/2), str, color="w", ha="center", va="center", fontweight="normal", fontsize=8)

  ax.set(xlabel=r'$p_t$ [GeV]', ylabel = r'$|\eta|$')
  ax.set_xscale('log')
  plt.savefig(output_figures)

  return sfhist, err_sfhist

def save_hist_schemav2(hist, desc, filename):
  cset = correctionlib.schemav2.CorrectionSet(
      schema_version=2,
      description=desc,
      corrections=[
          hist
      ],
  )
  with open(filename, "w") as fout:
      fout.write(cset.json(exclude_unset=True))
  return

def Corr_to_vec(BTag_eff, x_centers, y_centers):
    vec = []
    for x in np.arange(len(x_centers)):
        vec.append([])
        for y in np.arange(len(y_centers)):
            vec[x].append(BTag_eff.evaluate(x_centers[x], y_centers[y]))
    return np.array(vec)