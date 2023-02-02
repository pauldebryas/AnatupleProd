#!/usr/bin/env python
from coffea import processor
from coffea.nanoevents import NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = True
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
import collections

from CoffeaAnalysis.HNLAnalysis.helpers import files_from_path
from CoffeaAnalysis.stitching.DY.CountEventsDYJets import CountEventsNJetsPtZ

# parameters ----------------------------------------------------------------------------
file_dir = '/eos/user/p/pdebryas/HNL/nanoV10/Run2_2018/'
DIR_PATH = '/afs/cern.ch/user/p/pdebryas/HNL_analysis/NewDir/My_HNLTauPrompt'

inclusive_samples = ['DYJetsToLL_M-50']

exclusive_samples_NJets = [
    'DYJetsToLL_0J',
    'DYJetsToLL_1J',
    'DYJetsToLL_2J']

exclusive_samples_PtZ = [
    'DYJetsToLL_LHEFilterPtZ-0To50',
    'DYJetsToLL_LHEFilterPtZ-50To100',
    'DYJetsToLL_LHEFilterPtZ-100To250',
    'DYJetsToLL_LHEFilterPtZ-250To400',
    'DYJetsToLL_LHEFilterPtZ-400To650',
    'DYJetsToLL_LHEFilterPtZ-650ToInf']


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
        
    if P_i_j['DYJetsToLL_M-50'][i] == 0:
        #PS not populated by events
        s.append(0)
    else:
        s.append( (P_i_j['DYJetsToLL_M-50'][i]*N_sumw['DYJetsToLL_M-50']) / (P_i_j['DYJetsToLL_M-50'][i]*N_sumw['DYJetsToLL_M-50'] + sum(np.array(P_j)*np.array(N_j))) )


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
with open(f'{DIR_PATH}/CoffeaAnalysis/stitching/stitching_weights_2D_DYtoLL.json', "w") as outfile:
    outfile.write(json_object)