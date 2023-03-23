## How to run CoffeaAnalysis 

##  Require

- stitching files for DY and WJets
- counter.pkl which save the original sumw (before skimming) of MC samples (backgrounds and HNL signal) (for the reweighting of the events)

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

Tasks can be interdependent. To not mix up jobs, specify the same "version" parameter for all the tasks (here we use "nanoV10" used to produce nanoAOD root files).
For now, we used only 2018 samples.

Selection may involve: "tag" parameter is used to differenciate between selection.

Here an example for ttm channel. Same thing for tte tee tmm tem.

- You can print task dependencies with
```shell
law run Analysis --version nanoV10 --periods 2018 --tag TEST --Analysis-channel ttm --print-deps -1
```

- You can print task status of the task with
```shell
law run Analysis --version nanoV10 --periods 2018 --tag TEST --Analysis-channel ttm --print-status -1
```

- Run a single branch locally (useful for debugging)
```shell
law run Analysis --version nanoV10 --periods 2018 --tag TEST --Analysis-channel ttm --Analysis-workflow local --branch 41
```

- Run all branches with HTcondor
```shell
law run Analysis --version nanoV10 --periods 2018 --tag TEST --Analysis-channel ttm
```