#!/usr/bin/env python
from coffea import processor
from coffea.nanoevents import NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False
import numpy as np
import json
import collections
import os

from CoffeaAnalysis.stitching.DY.CountEventsDYJets import CountEventsNJetsPtZ
from CoffeaAnalysis.task_helpers import files_from_path

# parameters ----------------------------------------------------------------------------
period = '2018'
# ---------------------------------------------------------------------------------------
    
file_dir = f'{os.getenv("CENTRAL_STORAGE_NANOAOD")}/Run2_{period}/'

inclusive_samples = ['DYJetsToLL_M-50-amcatnloFXFX']

exclusive_samples_NJets = [
    'DYJetsToLL_0J-amcatnloFXFX',
    'DYJetsToLL_1J-amcatnloFXFX',
    'DYJetsToLL_2J-amcatnloFXFX']

exclusive_samples_PtZ = [
    'DYJetsToLL_LHEFilterPtZ-0To50-amcatnloFXFX',
    'DYJetsToLL_LHEFilterPtZ-50To100-amcatnloFXFX',
    'DYJetsToLL_LHEFilterPtZ-100To250-amcatnloFXFX',
    'DYJetsToLL_LHEFilterPtZ-250To400-amcatnloFXFX',
    'DYJetsToLL_LHEFilterPtZ-400To650-amcatnloFXFX',
    'DYJetsToLL_LHEFilterPtZ-650ToInf-amcatnloFXFX']


# look for number of events ----------------------------------------------------------------
exclusive_samples = exclusive_samples_NJets + exclusive_samples_PtZ
all_samples = inclusive_samples+exclusive_samples
nPtZbin = len(CountEventsNJetsPtZ.get_PtZ_bins())

PS_regions = []
for NJets in CountEventsNJetsPtZ.get_NJets_bins():
    for PtZbin in CountEventsNJetsPtZ.get_PtZ_bins():
        PS_regions.append(f'sumw_PtZ-{PtZbin}_Jets-{NJets}')

samples = {}
for element in all_samples:
    samples[element] = files_from_path(file_dir+element)

'''
EventsNotSelected_counter = processor.run_uproot_job(
    samples,
    'EventsNotSelected',
    CountEventsNJetsPtZ(),
    processor.iterative_executor,
    {"schema": NanoAODSchema, 'workers': 6},
)

EventsSelected_counter = processor.run_uproot_job(
    samples,
    'Events',
    CountEventsNJetsPtZ(),
    processor.iterative_executor,
    {"schema": NanoAODSchema, 'workers': 6},
)

# sumary sumw -----------------------------------------------------------------------------
print(f'| {"Samples":<35}', end='')
print(f'| {"selected events":<20}', end='')
print(f'| {"not selected events":<20}', end='')
print(f'| {"selected + notSelected":<20}', end='\n')

print(u'\u2500' * 100)
for sample in list(samples):
    print(f'| {sample:<35}', end='')
    print(f"| {round(EventsSelected_counter['sumw'][sample],1):.3e}           ", end="")
    print(f"| {round(EventsNotSelected_counter['sumw'][sample],1):.3e}           ", end='')
    print(f"| {round(EventsNotSelected_counter['sumw'][sample]+ EventsSelected_counter['sumw'][sample],1):.7e}         ", end='\n')

    print(u'\u2500' * 100)


# compute probability ----------------------------------------------------------------------
P_i_j = {}

for sample in list(samples):
    vector = []
    for PS_region in PS_regions:
        vector.append((EventsNotSelected_counter[PS_region][sample]+ EventsSelected_counter[PS_region][sample])/(EventsNotSelected_counter['sumw'][sample]+EventsSelected_counter['sumw'][sample]))
    P_i_j[sample] = np.array(vector)


#sumary probability  ------------------------------------------------------------------------
for NJets in CountEventsNJetsPtZ.get_NJets_bins():
    print('')
    print('NJets =' + NJets)
    print(f'| {"Samples":<35}', end='')
    for PtZbin in CountEventsNJetsPtZ.get_PtZ_bins():
        print(f'| {"PtZ= "+PtZbin+" ":<13}', end='')
    print(' ', end='\n')
    print(u'\u2500' * 140)

    for sample in list(samples):
        print(f'| {sample:<35}', end='')
        for i in np.arange(nPtZbin):
            print(f'| {round(P_i_j[sample][nPtZbin*int(NJets)+i],10):<12} ', end='')
        print(' ', end='\n')

        print(u'\u2500' * 140)

'''
#to be deleted
EventsSelected_counter = {}
EventsSelected_counter['sumw'] = {}
EventsSelected_counter['sumw']['DYJetsToLL_M-50-amcatnloFXFX'] = 5.677e+07
EventsSelected_counter['sumw']['DYJetsToLL_0J-amcatnloFXFX'] = 2.799e+07
EventsSelected_counter['sumw']['DYJetsToLL_1J-amcatnloFXFX'] = 2.260e+07
EventsSelected_counter['sumw']['DYJetsToLL_2J-amcatnloFXFX'] = 7.879e+06
EventsSelected_counter['sumw']['DYJetsToLL_LHEFilterPtZ-0To50-amcatnloFXFX'] = 4.852e+07
EventsSelected_counter['sumw']['DYJetsToLL_LHEFilterPtZ-50To100-amcatnloFXFX'] = 2.887e+07
EventsSelected_counter['sumw']['DYJetsToLL_LHEFilterPtZ-100To250-amcatnloFXFX'] = 2.249e+07
EventsSelected_counter['sumw']['DYJetsToLL_LHEFilterPtZ-250To400-amcatnloFXFX'] = 8.829e+06
EventsSelected_counter['sumw']['DYJetsToLL_LHEFilterPtZ-400To650-amcatnloFXFX'] = 1.601e+06
EventsSelected_counter['sumw']['DYJetsToLL_LHEFilterPtZ-650ToInf-amcatnloFXFX'] = 1.741e+06

EventsNotSelected_counter  = {}
EventsNotSelected_counter['sumw'] = {}
EventsNotSelected_counter['sumw']['DYJetsToLL_M-50-amcatnloFXFX'] = 7.555e+07
EventsNotSelected_counter['sumw']['DYJetsToLL_0J-amcatnloFXFX'] = 4.150e+07
EventsNotSelected_counter['sumw']['DYJetsToLL_1J-amcatnloFXFX'] = 2.075e+07
EventsNotSelected_counter['sumw']['DYJetsToLL_2J-amcatnloFXFX'] = 5.914e+06
EventsNotSelected_counter['sumw']['DYJetsToLL_LHEFilterPtZ-0To50-amcatnloFXFX'] = 5.280e+07
EventsNotSelected_counter['sumw']['DYJetsToLL_LHEFilterPtZ-50To100-amcatnloFXFX'] = 2.223e+07
EventsNotSelected_counter['sumw']['DYJetsToLL_LHEFilterPtZ-100To250-amcatnloFXFX'] = 9.628e+06
EventsNotSelected_counter['sumw']['DYJetsToLL_LHEFilterPtZ-250To400-amcatnloFXFX'] = 1.325e+06
EventsNotSelected_counter['sumw']['DYJetsToLL_LHEFilterPtZ-400To650-amcatnloFXFX'] = 1.389e+05
EventsNotSelected_counter['sumw']['DYJetsToLL_LHEFilterPtZ-650ToInf-amcatnloFXFX'] = 1.297e+05

P_i_j = {} 
P_i_j['DYJetsToLL_M-50-amcatnloFXFX'] = np.array([0.6889460024 , 0.1049187564 , 8.45513e-05  , 2.7206e-06   , 2.27e-08     , 7.6e-09      , 0.0 , 0.0          , 0.1000490674 , 0.0421108876 , 0.0067809735 , 0.0001650368 , 1.68604e-05  , 1.3754e-06 , 0.0          , 0.0277026187 , 0.0201594061 , 0.0085421394 , 0.0004451187 , 6.78874e-05  , 6.5673e-06   ])
P_i_j['DYJetsToLL_0J-amcatnloFXFX'] =   np.array([0.8671915858 , 0.1326997424 , 0.0001053046 , 3.3384e-06   , 2.88e-08     , 0.0          , 0.0 , 0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 0.0        , 0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 0.])
P_i_j['DYJetsToLL_1J-amcatnloFXFX'] =   np.array([0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 0.0 , 0.0          , 0.67097498   , 0.2821493034 , 0.0456462221 , 0.0011033339 , 0.0001186403 , 7.5203e-06 , 0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 0.0])
P_i_j['DYJetsToLL_2J-amcatnloFXFX'] =   np.array([0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 0.0 , 0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 0.0        , 0.0          , 0.4881635542 , 0.3534141813 , 0.1494513265 , 0.0076951234 , 0.0011545984 , 0.0001212162])
P_i_j['DYJetsToLL_LHEFilterPtZ-0To50-amcatnloFXFX'] =    np.array([0.0          , 0.4509142583 , 0.0          , 0.0          , 0.0          , 0.0          , 0.0                , 0.0          , 0.4298983647 , 2.96e-08     , 0.0          , 0.0          , 0.0          , 0.0 , 0.0          , 0.1191873573 , -9.9e-09     , 0.0          , 0.0          , 0.0          , 0.0])
P_i_j['DYJetsToLL_LHEFilterPtZ-50To100-amcatnloFXFX'] =  np.array([0.0          , 0.0          , 0.0013584307 , 0.0          , 0.0          , 0.0          , 0.0                , 0.0          , 0.0          , 0.6787180257 , 3.91e-08     , 0.0          , 0.0          , 0.0 , 0.0          , 0.0          , 0.3199234458 , 5.87e-08     , 0.0          , 0.0          , 0.0])
P_i_j['DYJetsToLL_LHEFilterPtZ-100To250-amcatnloFXFX'] = np.array([0.0          , 0.0          , 0.0          , 0.0001724025 , 0.0          , 0.0          , 0.0                , 0.0          , 0.0          , 0.0          , 0.4525559832 , 0.0          , 0.0          , 0.0 , 0.0          , 0.0          , 0.0          , 0.5472716144 , 0.0          , 0.0          , 0.0])
P_i_j['DYJetsToLL_LHEFilterPtZ-250To400-amcatnloFXFX'] = np.array([0.0          , 0.0          , 0.0          , 0.0          , 5.51531e-05  , 0.0          , 0.0                , 0.0          , 0.0          , 0.0          , 0.0          , 0.2814048595 , 0.0          , 0.0 , 0.0          , 0.0          , 0.0          , 0.0          , 0.7185399874 , 0.0          , 0.0])
P_i_j['DYJetsToLL_LHEFilterPtZ-400To650-amcatnloFXFX'] = np.array([0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 2.41357e-05  , 0.0                , 0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 0.2132078584 , 0.0 , 0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 0.7867680059 , 0.0])
P_i_j['DYJetsToLL_LHEFilterPtZ-650ToInf-amcatnloFXFX'] = np.array([0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 1.97807e-05        , 0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 0.1578668243 , 0.0 , 0.0          , 0.0          , 0.0          , 0.0          , 0.0          , 0.8421133949])
#---------------------------------------------------------------

# compute stitching weights  ---------------------------------------------------------------
N_sumw = {}
for sample in list(samples):
    N_sumw[sample] = EventsNotSelected_counter['sumw'][sample]+ EventsSelected_counter['sumw'][sample]

N_j = []
for exclusive_sample in exclusive_samples:
    N_j.append(N_sumw[exclusive_sample])

s = []
for i in range(len(PS_regions)):
    P_j = []
    for exclusive_sample in exclusive_samples:
        P_j.append(P_i_j[exclusive_sample][i])
        
    if P_i_j['DYJetsToLL_M-50-amcatnloFXFX'][i] == 0:
        #PS not populated by events
        s.append(0)
    else:
        s.append( (P_i_j['DYJetsToLL_M-50-amcatnloFXFX'][i]*N_sumw['DYJetsToLL_M-50-amcatnloFXFX']) / (P_i_j['DYJetsToLL_M-50-amcatnloFXFX'][i]*N_sumw['DYJetsToLL_M-50-amcatnloFXFX'] + sum(np.array(P_j)*np.array(N_j))) )


# sumary stitching weights ----------------------------------------------------------------- 
print('')
print(f'| {"                 ":<10}', end='')
for PtZbin in CountEventsNJetsPtZ.get_PtZ_bins():
    print(f'| {"PtZ="+PtZbin+" ":<13}', end='')
print(' ', end='\n')
print(u'\u2500' * 170)

for NJets in CountEventsNJetsPtZ.get_NJets_bins():
    print(f'| NJets= {NJets:<10}', end='')
    for i in np.arange(nPtZbin):
        print(f'| {round(s[nPtZbin*int(NJets)+i],6):<12} ', end='')
    print(' ', end='\n')

    print(u'\u2500' * 170)

 
# SW to be written ------------------------------------------------------------------------
stitching_weights = collections.defaultdict(dict)
i = 0
for NJets in CountEventsNJetsPtZ.get_NJets_bins():
    j=0
    for PtZbin in CountEventsNJetsPtZ.get_PtZ_bins():
        stitching_weights[f'NJets={NJets}'][f'PtZ={PtZbin}'] = s[nPtZbin*i+j]
        j=j+1
    i= i+1

# Serializing json
json_object = json.dumps(stitching_weights, indent=4)
 
# Writing to sample.json
output_results_folder = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/stitching/data/{period}/'
if not os.path.exists(output_results_folder):
    os.makedirs(output_results_folder)  

with open(output_results_folder + 'stitching_weights_2D_DYtoLL.json', "w") as outfile:
    outfile.write(json_object)
