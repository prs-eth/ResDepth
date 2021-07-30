#!/usr/bin/env bash


function download_demo() {
    if [ ! -d "demo" ]; then
        mkdir -p "demo"
    fi
    cd demo 

    url="https://share.phys.ethz.ch/~pf/stuckercdata/resdepth/"
   
    model_file="demo.tar" 
    
    wget --no-check-certificate --show-progress "$url$model_file"
    tar -xf "$model_file"
    rm "$model_file"
    cd ../
}


download_demo;