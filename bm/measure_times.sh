#!/bin/bash

if [ -z $dock_home ]
then
    echo "not found envoriment variable : dock_home"
fi

set -ex
declare -a stringArray
stringArray=("2VXT" "3VLB" "2A1A" "2GTP" "2X9A" "1RKE" "3LVK" "4GAM" "4JCV" "4LW4")
HOME=$dock_home/bm/
STEPS=100
# dock_home=/share/home/mengxiangyu/sparkledock


export LD_LIBRARY_PATH=$dock_home/lib:$LD_LIBRARY_PATH
export NUM_THREADS=32
export USE_CUDA=1


for str in "${stringArray[@]}"; do

echo "prepare $str" >> ${HOME}measure.txt

COMPLEX=$str

echo ${HOME}${COMPLEX}
cd ${HOME}${COMPLEX}
# # Setup
rm -rf lightdock* init/ swarm_*
# cp ../${COMPLEX}_A_noh.pdb ../${COMPLEX}_B_noh.pdb .

# rm -rf swarm_*
# rm -rf clustered

# ncu --set roofline -f -o opt_cuda_base_${COMPLEX}  $dock_home/bin/lightdock -f setup.json -s $STEPS -l 1 
mpirun -np 1 $dock_home/bin/lightdock -f setup.json -s $STEPS -l 1 >> ${HOME}measure.txt

done