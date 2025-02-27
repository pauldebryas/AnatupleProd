#!/usr/bin/env bash

action() {
    # determine the directory of this file
    local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"

    export PYTHONPATH="$this_dir:$PYTHONPATH"
    export LAW_HOME="$this_dir/.law"
    export LAW_CONFIG_FILE="$this_dir/config/law.cfg"

    export ANALYSIS_PATH="$this_dir"
    export ANALYSIS_DATA_PATH="$ANALYSIS_PATH/data"
    export X509_USER_PROXY="$ANALYSIS_DATA_PATH/voms.proxy"
    export CENTRAL_STORAGE_ANATUPLE="/eos/user/p/pdebryas/HNL_LLFF"
    export CENTRAL_STORAGE_NANOAOD="/eos/cms/store/group/phys_higgs/HLepRare/HTT_skim_v1"

    export PATH=$PATH:$HOME/.local/bin:$ANALYSIS_PATH/scripts
    
    source /afs/cern.ch/work/p/pdebryas/AnalEnv/bin/activate
    source "$( law completion )" ""
}
action