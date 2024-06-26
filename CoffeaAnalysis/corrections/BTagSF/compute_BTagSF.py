#import ROOT
import os
import yaml
from BTagSF.helpers import load_ntuples, get_correction_central, make_hist_sf, save_hist_schemav2
import numpy as np
import awkward as ak
import correctionlib.convert
import ROOT
import argparse
#computing BTag efficiency as function of pt/eta of the jet and per Jet flavor

# example: python BTagSF/compute_BTagSF.py --channel tem --JetFlavor 0

#parameters -----------------------------------------------------------------------------------------------------
#period = period of data taking
period = '2017'
#tag = tag used for anatuple production
tag = 'WithCorrections'
#nbins = number of bins (same for pt and eta) for BTag efficiency estimation.
nbins = 5

descr = {
    '0':'light quarks (d,u,s), gluon or undefined',
    '4':'c quark',
    '5':'b quark'
}

parser = argparse.ArgumentParser(description="Please provide channel and Jet_flavor as arguments.")
parser.add_argument("--channel", type=str, help="channel: can be tte, tee, ttm, tmm or tem")
parser.add_argument("--JetFlavor", type=str, help="JetFlavor: can be 0, 4 or 5")
args = parser.parse_args()

if args.channel is None or args.JetFlavor is None:
    raise("Please provide both arguments.")

if args.JetFlavor not in ['0','4','5']:
    raise('Invalide value for Jet Flavor')

if args.channel not in ['ttm','tmm','tte', 'tee', 'tem']:
    raise('Invalide value for Jet Flavor')

# Print the provided arguments
print("channel:", args.channel)
print("JetFlavor:", args.JetFlavor)

#channel = channel for BTag efficiency estimation
channel = args.channel
Jet_flavor = args.JetFlavor 
# 0 is light quarks (d,u,s), gluon, undefined
# 4 is C quark
# 5 is B quark
#----------------------------------------------------------------------------------------------------------------


# btag wp
cset = get_correction_central(period)
#loose WP value
deepJet_wp_value = cset["deepJet_wp_values"].evaluate("L")

output_results_folder = os.path.join(os.getenv("ANALYSIS_PATH"),f'CoffeaAnalysis/corrections/BTagSF/data/{period}/') 
if not os.path.exists(output_results_folder):
    os.makedirs(output_results_folder)     

output_results_file = os.path.join(output_results_folder, f'btagEff_{channel}_{Jet_flavor}.root')

output_json_path = os.path.join(os.getenv("ANALYSIS_PATH"),f'CoffeaAnalysis/corrections/BTagSF/corr_schema/{period}/') 
if not os.path.exists(output_json_path):
    os.makedirs(output_json_path)     

output_json_file = os.path.join(output_json_path, f'btagEff_{channel}_{Jet_flavor}.json')

output_figures_path = os.path.join(os.getenv("ANALYSIS_PATH"),f'CoffeaAnalysis/corrections/BTagSF/figures/{period}/Jet_flavor_{Jet_flavor}/')
if not os.path.exists(output_figures_path):
    os.makedirs(output_figures_path)    

output_figure_file = os.path.join(output_figures_path, f'btagEff_{channel}.pdf')

# load input samples
config_file = os.path.join(os.getenv("ANALYSIS_PATH"), 'config', f'samples_{period}.yaml')
with open(config_file, 'r') as f:
    samples = yaml.safe_load(f)

inputs = {}
for sample_name in sorted(samples.keys()):
    sampleType = samples[sample_name].get('sampleType', None)
    if (sampleType != 'data') & ('HNL' not in sample_name):
        inputs[sample_name] = sample_name+'_anatuple.root'

# Create a ROOT file
root_file = ROOT.TFile(output_results_file, "RECREATE")

input_dir = os.path.join(os.getenv("CENTRAL_STORAGE_ANATUPLE"),'anatuple', period, tag, channel, 'anatuple')
input_files = {}
for elem in inputs.keys():
    file = os.path.join(input_dir, inputs[elem])
    if os.path.isfile(file) == False:
        if file[:4] not in ['WWTo', 'WZTo','ZZTo']:
            print('WARNING: ' + file + ' is missing')
    else:
        input_files[elem] = file

print(f'--Loading anatuples--')
branches = load_ntuples(input_files)

#compute BTag efficiency
print(f'--processing Jet_flavor {Jet_flavor}--')
Jet_pt_den = []
Jet_eta_den = []
Jet_pt_num = []
Jet_eta_num = []
for MCsample in branches.keys():
    print(f'----{MCsample}')
    mask_den = branches[MCsample]['hadronFlavour'] == int(Jet_flavor)
    mask_WP = branches[MCsample]['btagDeepFlavB'] >= deepJet_wp_value 
    Jet_pt_den.append(ak.concatenate(branches[MCsample]['pt'][mask_den]))
    Jet_eta_den.append(ak.concatenate(branches[MCsample]['eta'][mask_den]))
    Jet_pt_num.append(ak.concatenate(branches[MCsample]['pt'][mask_den & mask_WP]))
    Jet_eta_num.append(ak.concatenate(branches[MCsample]['eta'][mask_den & mask_WP]))

Jet_pt_den = np.array(ak.concatenate(Jet_pt_den))
Jet_eta_den = np.array(ak.concatenate(Jet_eta_den))
Jet_pt_num = np.array(ak.concatenate(Jet_pt_num))
Jet_eta_num = np.array(ak.concatenate(Jet_eta_num))

sfhist, err_sfhist = make_hist_sf(Jet_pt_den, Jet_eta_den, Jet_pt_num, Jet_eta_num, nbins, output_figure_file)

fake_rate = correctionlib.convert.from_histogram(sfhist)
fake_rate_err = correctionlib.convert.from_histogram(err_sfhist)

# set overflow bins behavior (default is to raise an error when out of bounds)
fake_rate.data.flow = "clamp"
fake_rate_err.data.flow = "clamp"

save_hist_schemav2(fake_rate, f"fake rate", output_json_file)
output_err_json_file = os.path.join(output_json_path, f'btagEff_{channel}_{Jet_flavor}_err.json')
save_hist_schemav2(fake_rate_err, f"fake rate err", output_err_json_file)

# Create TH2D histogram
hist1 = ROOT.TH2D(f'jet_pt_eta_{Jet_flavor}', f'Selected jets with {descr[Jet_flavor]} flavor', 100, 0, np.max(Jet_pt_den), 100, 0,  np.max( abs(Jet_eta_den) ) )
hist1.FillN(len(Jet_pt_den), Jet_pt_den,  abs(Jet_eta_den), np.ones_like(Jet_pt_den))
hist1.Write()

hist2 = ROOT.TH2D(f'jet_pt_eta_{Jet_flavor}_Loose', f'Selected jets with {descr[Jet_flavor]} flavor and Loose btag WP', 100, 0, np.max(Jet_pt_num), 100, 0,  np.max( abs(Jet_eta_num)))
hist2.FillN(len(Jet_pt_num), Jet_pt_num,  abs(Jet_eta_num), np.ones_like(Jet_pt_num))
hist2.Write()

# Close the ROOT file
root_file.Close()

# to do:
#    - hadd root file together
#    - compute BTagEff for channel combined