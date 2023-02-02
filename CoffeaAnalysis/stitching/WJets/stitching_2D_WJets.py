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
from CoffeaAnalysis.stitching.WJets.CountEventsWJets import CountEventsNJetsHT

# parameters ----------------------------------------------------------------------------
file_dir = '/eos/user/p/pdebryas/HNL/nanoV10/Run2_2018/'
DIR_PATH = '/afs/cern.ch/user/p/pdebryas/HNL_analysis/NewDir/My_HNLTauPrompt'

inclusive_samples = ['WJetsToLNu']

exclusive_samples_NJets = [
    'W1JetsToLNu',
    'W2JetsToLNu',
    'W3JetsToLNu',
    'W4JetsToLNu']

exclusive_samples_HT = [
    'WJetsToLNu_HT-70To100',
    'WJetsToLNu_HT-100To200',
    'WJetsToLNu_HT-200To400',
    'WJetsToLNu_HT-400To600',
    'WJetsToLNu_HT-600To800',
    'WJetsToLNu_HT-800To1200',
    'WJetsToLNu_HT-1200To2500',
    'WJetsToLNu_HT-2500ToInf']



# look for number of events ----------------------------------------------------------------
exclusive_samples = exclusive_samples_NJets + exclusive_samples_HT
all_samples = inclusive_samples+exclusive_samples
nHTbin = len(CountEventsNJetsHT.get_HT_bins())

PS_regions = []
for NJets in CountEventsNJetsHT.get_NJets_bins():
    for HTbin in CountEventsNJetsHT.get_HT_bins():
        PS_regions.append(f'sumw_HT-{HTbin}_Jets-{NJets}')

samples = {}
for element in all_samples:
    samples[element] = files_from_path(file_dir+element)

EventsNotSelected_counter = processor.run_uproot_job(
    samples,
    'EventsNotSelected',
    CountEventsNJetsHT(),
    processor.iterative_executor,
    {"schema": NanoAODSchema, 'workers': 6},
)

EventsSelected_counter = processor.run_uproot_job(
    samples,
    'Events',
    CountEventsNJetsHT(),
    processor.iterative_executor,
    {"schema": NanoAODSchema, 'workers': 6},
)


# sumary sumw -----------------------------------------------------------------------------
print(f'| {"Samples":<25}', end='')
print(f'| {"selected events":<20}', end='')
print(f'| {"not selected events":<20}', end='')
print(f'| {"selected + notSelected":<20}', end='\n')

print(u'\u2500' * 100)
for sample in list(samples):
    print(f'| {sample:<25}', end='')
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
for NJets in CountEventsNJetsHT.get_NJets_bins():
    print('')
    print('NJets =' + NJets)
    print(f'| {"Samples":<25}', end='')
    for HTbin in CountEventsNJetsHT.get_HT_bins():
        print(f'| {"HT= "+HTbin+" ":<13}', end='')
    print(' ', end='\n')
    print(u'\u2500' * 180)

    for sample in list(samples):
        print(f'| {sample:<25}', end='')
        for i in np.arange(nHTbin):
            print(f'| {round(P_i_j[sample][nHTbin*int(NJets)+i],10):<12} ', end='')
        print(' ', end='\n')

        print(u'\u2500' * 180)


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
        
    if P_i_j['WJetsToLNu'][i] == 0:
        #PS not populated by events
        s.append(0)
    else:
        s.append( (P_i_j['WJetsToLNu'][i]*N_sumw['WJetsToLNu']) / (P_i_j['WJetsToLNu'][i]*N_sumw['WJetsToLNu'] + sum(np.array(P_j)*np.array(N_j))) )


# sumary stitching weights ----------------------------------------------------------------- 
print('')
print(f'| {"                 ":<10}', end='')
for HTbin in CountEventsNJetsHT.get_HT_bins():
    print(f'| {"HT="+HTbin+" ":<13}', end='')
print(' ', end='\n')
print(u'\u2500' * 170)

for NJets in CountEventsNJetsHT.get_NJets_bins():
    print(f'| NJets= {NJets:<10}', end='')
    for i in np.arange(nHTbin):
        print(f'| {round(s[nHTbin*int(NJets)+i],6):<12} ', end='')
    print(' ', end='\n')

    print(u'\u2500' * 170)

 
# SW to be written ------------------------------------------------------------------------
stitching_weights = collections.defaultdict(dict)
i = 0
for NJets in CountEventsNJetsHT.get_NJets_bins():
    j=0
    for HTbin in CountEventsNJetsHT.get_HT_bins():
        stitching_weights[f'NJets={NJets}'][f'HT={HTbin}'] = s[nHTbin*i+j]
        j=j+1
    i= i+1

# Serializing json
json_object = json.dumps(stitching_weights, indent=4)
 
# Writing to sample.json
with open(f'{DIR_PATH}/CoffeaAnalysis/stitching/stitching_weights_2D_WJetsToLNu.json', "w") as outfile:
    outfile.write(json_object)