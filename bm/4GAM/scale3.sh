#!/bin/bash
#DSUB -n job_test
#DSUB -N 1
#DSUB -A root.l6eub2ic
#DSUB -R "cpu=128;gpu=4;mem=240000"	
#DSUB -oo scale3_%J.out					
#DSUB -eo scale3_%J.err			

export LD_LIBRARY_PATH=/home/l6eub2ic/whcs-share31/mengxiangyu/sparkledock/lib:$LD_LIBRARY_PATH
export NUM_THREADS=32
export USE_CUDA=1
STEPS=100
dock_home=/home/l6eub2ic/whcs-share31/mengxiangyu/sparkledock
rm -rf swarm_*
rm -rf clustered

mpirun -np 4 $dock_home/bin/lightdock -f setup.json -s $STEPS -l 1 

rm -rf swarm_*
rm -rf clustered