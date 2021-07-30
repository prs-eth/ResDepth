#!/usr/bin/env bash


function download_models() {
    if [ ! -d "logs" ]; then
        mkdir -p "logs"
    fi
    cd logs 

    url="https://share.phys.ethz.ch/~pf/stuckercdata/resdepth/"
   
    model_file="pretrained_models_ablations.tar" 
    
    wget --no-check-certificate --show-progress "$url$model_file"
    tar -xf "$model_file"
    rm "$model_file"
    cd ../
}


download_models;