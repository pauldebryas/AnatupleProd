import yaml
import os
import json

def load_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
    return data

def save_yaml(file_path, data):
    try:
        with open(file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
        print(f"YAML file saved: {file_path}")
    except Exception as e:
        print(f"Error saving YAML file: {e}")

def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
    return data

def check_folders(path):
    list_folder = {}
    try:
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path):
                list_folder[folder] = folder_path
    except Exception as e:
        print(f"Error checking folders: {e}")
    return list_folder

# parameters
period = '2018'
config_yaml_file = f'config/samples_{period}_HNL.yaml'
output_config_yaml_file = f'config/samples_{period}.yaml'
NanoFiles_path = f'/eos/cms/store/group/phys_higgs/HLepRare/HTT_skim_v1/Run2_{period}/'

folder_dict = check_folders(NanoFiles_path)
config_file_dict = load_yaml(config_yaml_file)

common_sample = {}
for item_HNLprod in list(config_file_dict.keys()):
    item_HTTprod = item_HNLprod

    #some samples don't have the same name
    if item_HNLprod[0:11]=='DYJetsToLL_':
        item_HTTprod = item_HNLprod + '-amcatnloFXFX'
    if item_HNLprod == 'ST_tW_antitop_5f_inclusiveDecays':
        item_HTTprod = 'ST_tW_antitop_5f_InclusiveDecays'
    if item_HNLprod == 'ST_tW_top_5f_inclusiveDecays':
        item_HTTprod = 'ST_tW_top_5f_InclusiveDecays'
    if item_HNLprod in ['WWW', 'WWZ']:
        item_HTTprod = item_HNLprod + '_4F'
    if item_HNLprod in ['WminusHToTauTau', 'WplusHToTauTau', 'GluGluHToTauTau', 'ZHToTauTau', 'ttHToTauTau']:
        item_HTTprod = item_HNLprod + '_M125'
    if item_HNLprod == 'EWKWMinus2Jets_WToLNu_M-50':
        item_HTTprod = 'EWK_WminusToLNu'
    if item_HNLprod == 'EWKWPlus2Jets_WToLNu_M-50':
        item_HTTprod = 'EWK_WplusToLNu'
    if item_HNLprod == 'EWKZ2Jets_ZToLL_M-50':
        item_HTTprod = 'EWK_ZTo2L'
    if item_HNLprod.startswith('EGamma_'):
        item_HTTprod = item_HNLprod[0:7] + 'Run' + item_HNLprod[7:]
    if item_HNLprod.startswith('SingleMuon_'):
        item_HTTprod = item_HNLprod[0:11] + 'Run' + item_HNLprod[11:]
    if item_HNLprod.startswith('Tau_'):
        item_HTTprod = item_HNLprod[0:4] + 'Run' + item_HNLprod[4:]

    if item_HTTprod in list(folder_dict.keys()):
        prodReport_json = os.path.join(folder_dict[item_HTTprod], 'prodReport_nanoHTT.json')
        prodReport = load_json(prodReport_json)
        miniAODpathHTTProd = prodReport["inputDataset"]
        miniAODpathHNLProd = config_file_dict[item_HNLprod]['miniAOD']
        if miniAODpathHTTProd == miniAODpathHNLProd:
            common_sample[item_HNLprod] = {}
            common_sample[item_HNLprod]['HNLprodName'] = item_HNLprod
            common_sample[item_HNLprod]['miniAOD'] = config_file_dict[item_HNLprod]['miniAOD']
            common_sample[item_HNLprod]['nanoAOD'] = config_file_dict[item_HNLprod]['nanoAOD']
            common_sample[item_HNLprod]['sampleType'] = config_file_dict[item_HNLprod]['sampleType']
            if config_file_dict[item_HNLprod]['sampleType'] != 'data':
                common_sample[item_HNLprod]['PhysicsType'] = config_file_dict[item_HNLprod]['PhysicsType']
                common_sample[item_HNLprod]['crossSection'] = config_file_dict[item_HNLprod]['crossSection']

            common_sample[item_HNLprod]['HTTprodName'] = item_HTTprod
            common_sample[item_HNLprod]['HTTprodPath'] = folder_dict[item_HTTprod]
            common_sample[item_HNLprod]['HTTprodReportFile'] = prodReport_json
        else:
            print('   not same miniAOD file:')
            print(f'      - HTT {miniAODpathHTTProd}')
            print(f'      - HNL {miniAODpathHNLProd}')
    else:
        if not item_HNLprod.startswith('HNL_'):
            print(f'   *** {item_HNLprod} not in HTT_skim_v1 ***')

save_yaml(output_config_yaml_file, common_sample)
