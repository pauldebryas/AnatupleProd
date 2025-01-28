import os
import yaml
from run_tools.sh_tools import sh_call
import json
import csv

#produce files where run number of data samples are stored
#parameters
year = '2016_HIPM'
#----------------------------------------------------

# produce dir if doesn't exist called run_Data_{year}
output_folder = os.path.join(os.getenv("ANALYSIS_PATH"), 'CoffeaAnalysis', 'luminosity', 'data', 'run_Data',f'{year}')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# fetch in config/samples_{year}.yaml all "  sampleType: data" key and save nanoAOD corresponding file
config_file = os.path.join(os.getenv("ANALYSIS_PATH"), 'config', f'samples_{year}.yaml')
with open(config_file, 'r') as f:
    samples = yaml.safe_load(f)

data_nanoAOD = {}
for sample_name in sorted(samples.keys()):
    sampleType = samples[sample_name].get('sampleType', None)
    if sampleType == 'data':
        NanoAODfile = samples[sample_name].get('nanoAOD', None)
        if NanoAODfile != None:
            data_nanoAOD[sample_name] = NanoAODfile
        else:
            print(f'missing NanoAOD in sample {sample_name}')
            raise

for data_name in sorted(data_nanoAOD.keys()):
    # Request on DAQ: f'run dataset={nanoAOD}' to get run number
    # use the following command to look at the Run number corresonding to the data sample: dasgoclient -json -query 'dataset= ...' 
    _, output = sh_call(['dasgoclient', '-json', '-query', f'run dataset={data_nanoAOD[data_name]}'], catch_stdout=True)
    output=json.loads(output)
    run_number = []
    for element in output:
        get_run = element.get('run', None)
        if get_run != None:
            run_number.append(get_run[0]['run_number'])

    # save in csv file run_{key}.csv
    print(f'Writing run_{data_name}.csv')
    #print(run_number)
    with open(os.path.join(output_folder, f'run_{data_name}.csv'), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # first line #Request on DAQ: f'run dataset={nanoAOD}'
        writer.writerow([f"#Request on DAQ: 'run dataset={data_nanoAOD[data_name]}'"])
        # 1 Run number per line
        for row in run_number:
            writer.writerow([str(row)])
