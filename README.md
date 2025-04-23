# HNL with tau analysis code

HNL-->3leptons analysis with hadronic tau using law workflow.

##  Requirements

You only need a CERN account, with access to DAS: https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookLocatingDataSamples

### Packages 

Here is a non-exhaustive list of the packages that I used:
- python3
- law
- coffea
- akward array
- root
- numpy
- matplotlib
- yaml

For simplicity, you can use the same conda environment as I, which is saved in a .yml file (located in /config).
For that:
- Make sure you have conda/miniconda installed: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
- Install and activate the environment that I named "HNL" with the command:
```shell
conda env create -f HNL_environment.yml
```

##  Running the analysis

- Clone this repositiry (to do once)
```shell
git clone git@github.com:cms-hnl/HNLTauPrompt.git
cd HNLTauPrompt/
```

- Setup the environment (to do at each login). Make sure you have the correct path to your repository in env.sh
```shell
source env.sh
```

- Create a proxy to setup your access to DAS
```shell
voms-proxy-init -voms cms -rfc -valid 192:00
```

Then you can use law to monitor and run the different tasks

### Law commands

- First you need to create an index file, too list all the tasks.
```shell
law index --verbose
```

- You can print task dependencies with
```shell
law run name_of_the_task --parameters1 param1 --parameters1 param2 ... --print-deps -1
```

- You can print task status of the task with
```shell
law run name_of_the_task --parameters1 param1 --parameters1 param2 ... --print-status -1
```

- Run task locally (useful for debugging). If the task have multiple branches you can use:
```shell
law run name_of_the_task --parameters1 param1 --parameters1 param2 ... --name_of_the_task-workflow local --branch index_of_the_branch
```

- Run task with HTcondor
```shell
law run name_of_the_task --parameters1 param1 --parameters1 param2 ...
```

- If you want to limit the number of jobs running simultaneously (EOS space management for skimed nanoAOD samples)
```shell
law run name_of_the_task --parameters1 param1 --parameters1 param2 ... --CreateNanoSkims-parallel-jobs 100
```

### Tasks

- CreateVomsProxy: creates a new proxy in order to access DAS: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideVomsFAQ
- CreateSamplesConfigFile: Create yaml file with the sample list that have been processed for a specific year. Output is in config/samples_{period}.yaml.
- RunCounter: save in a .pkl file for each year the original sum weights of the events (selected and not selected) in each samples. Needed for the Analysis task.
- RunAnalysis: skim HNL signal, background and data using Coffea (Columnar Object Framework For Effective Analysis). This task require additional argument --tag which will be the name of the folder where we store the results. 
The results are stored in two type of files (in CoffeaAnalysis/results/):
    - .pkl which save the sumw of the events through the cutflow 
    - .root which save the remaining events after all the cut with needed info for the events and the selected objects.

### Helpers to monitor results

- Produce 2 htlm files which contain information on the root file (doc= description of branches, size= exhaustive description of file's size) 
```shell
inspectNanoFile.py -d doc.html -s size.html /eos/user/p/pdebryas/HNL/v1/nanoAOD/2018/HNL_tau_M-1000.root
```

## Documentation
- Coffea: https://coffeateam.github.io/coffea/index.html
- pdgID of the particles: https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
- Akward array: https://awkward-array.readthedocs.io/en/latest/index.html
- law documentation: https://luigi.readthedocs.io/en/stable/#
- Example that demonstrates how to create law task workflows that run on the HTCondor batch system at CERN: https://github.com/riga/law/tree/master/examples/htcondor_at_cern
- Xsec: using DAS query for NanoAOD samples (https://cms-gen-dev.cern.ch/xsdb/) or using previous analysis (e.g https://github.com/hh-italian-group/hh-bbtautau/blob/master/Instruments/config/cross_section.cfg)
- Luminosity: In luminosity/run_Data there is the run number of all the runs in the data files (EGamma/SingleMuon/Tau) for 2018 and A/B/C/D area.
run2018_lumi.csv obtain at https://twiki.cern.ch/twiki/bin/view/CMS/TWikiLUM give the corresponding luminosity of all those run number, so you can compute the exact luminosity for all data samples. 
