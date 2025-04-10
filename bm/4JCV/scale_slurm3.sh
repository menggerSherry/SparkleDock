#!/bin/bash
#SBATCH --gpus=4
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 32
#SBATCH --output=./scale3_%j.out
	

export LD_LIBRARY_PATH=/home/l6eub2ic/whcs-share31/mengxiangyu/SparkleDock/lib:$LD_LIBRARY_PATH
export NUM_THREADS=32
export USE_CUDA=1
STEPS=100
dock_home=/home/l6eub2ic/whcs-share31/mengxiangyu/SparkleDock
rm -rf swarm_*
rm -rf clustered

mpirun -np 4 $dock_home/bin/lightdock -f setup.json -s $STEPS -l 1 

rm -rf swarm_*
rm -rf clustered