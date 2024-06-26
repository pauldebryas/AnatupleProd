## How to run the Analysis 

##  Require

- stitching files for DY and WJets. Change the period in the .py file.
```shell
python CoffeaAnalysis/stitching/DY/stitching_2D_DY.py
python CoffeaAnalysis/stitching/WJets/stitching_2D_WJets.py
```
Output will be saved in CoffeaAnalysis/stitching/data/ folder.

- luminosity: produce run_Data files. Change the period in the .py file. Need csv files (recipe available in LUMI POG Twiki).
```shell
python CoffeaAnalysis/luminosity/produce_run_Data.py
```

- counter.pkl which save the original sumw (before skimming) of MC samples (backgrounds and HNL signal) used for the reweighting of the events. Here is an example on how to run the task:
```shell
law run RunCounter --periods 2018
```

- corrections: add err histograms for the RECO and Trigger, needed to compute electron SFs variation.
```shell
python CoffeaAnalysis/corrections/Add_err_hist_RECOSF.py
python CoffeaAnalysis/corrections/Add_err_hist_triggerSF.py
```

- BTAG SFs. Save the root files and convert to json correction schema
```shell
python CoffeaAnalysis/corrections/BTagSF/compute_BTagSF.py
python CoffeaAnalysis/corrections/BTagSF/compute_meanBTagSF.py
```

##  Output

-anatuple file: root files with minimal information on the event and the 3 lepton candidates + bjets
-pickle file: files where are stored cutflow information of events during anatuple production

##  Running the code

- Setup the environment (to do at each login)
```shell
source env.sh
```
Make sure you have the correct path to your repository in env.sh

Then you can use law to monitor and run the different tasks

### Law commands

- First you need to create an index file, too list all the tasks.
```shell
law index --verbose
```

Here an example to run ttm channel for 2018 period:

- You can print task dependencies with
```shell
law run Analysis --periods 2018 --tag TAG --channel ttm --print-deps -1
```

- You can print task status of the task with
```shell
law run Analysis --periods 2018 --tag TAG --channel ttm --print-status -1
```

- Run a single branch locally (useful for debugging)
```shell
law run Analysis --periods 2018 --tag TAG --channel ttm --Analysis-workflow local --branch 41
```

- Run all branches with HTcondor
```shell
law run Analysis --periods 2018 --tag TAG --channel ttm
```