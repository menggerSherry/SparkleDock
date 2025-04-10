#!/bin/bash

if [ -z $dock_home ]
then
    echo "not found envoriment variable : dock_home"
fi

set -ex
declare -a stringArray
stringArray=("2VXT" "2X9A" "4GAM" "4JCV" "4LW4")
HOME=$dock_home/bm/
STEPS=100
# dock_home=/share/home/mengxiangyu/sparkledock
dconfig

export LD_LIBRARY_PATH=$dock_home/lib:$LD_LIBRARY_PATH
export NUM_THREADS=32
export USE_CUDA=1


for str in "${stringArray[@]}"; do

echo "prepare $str"

COMPLEX=$str

echo ${HOME}${COMPLEX}
cd ${HOME}${COMPLEX}
# # Setup
rm -rf lightdock* init/ swarm_*


dsub -s scale5.sh

done