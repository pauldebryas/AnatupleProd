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
    export CENTRAL_STORAGE="/eos/user/p/pdebryas/HNL"
    export ANALYSIS_BIG_DATA_PATH="$CENTRAL_STORAGE/tmp/$(whoami)/data"

    export PATH=$PATH:$HOME/.local/bin:$ANALYSIS_PATH/scripts
    #source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_101 x86_64-centos7-gcc8-opt

    PRIVATE_CONDA_INSTALL=/afs/cern.ch/work/p/pdebryas/miniconda3
    __conda_setup="$($PRIVATE_CONDA_INSTALL/bin/conda shell.${SHELL##*/} hook)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "$PRIVATE_CONDA_INSTALL/etc/profile.d/conda.sh" ]; then
            . "$PRIVATE_CONDA_INSTALL/etc/profile.d/conda.sh"
        else
            export PATH="$PRIVATE_CONDA_INSTALL/bin:$PATH"
        fi
    fi
    unset __conda_setup

    conda activate HNL

    source "$( law completion )" ""
}
action