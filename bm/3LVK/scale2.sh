#!/bin/bash
#SBATCH --gpus=2
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 32
#SBATCH --output=./scale2_%j.out

export LD_LIBRARY_PATH=/home/bingxing2/home/scx8jwu/mxy/sparkledock/lib:$LD_LIBRARY_PATH
export NUM_THREADS=32
export USE_CUDA=1
STEPS=100
dock_home=/home/bingxing2/home/scx8jwu/mxy/sparkledock
rm -rf swarm_*
rm -rf clustered

srun $dock_home/bin/lightdock -f setup.json -s $STEPS -l 1 

# rm -rf swarm_*
# rm -rf clustered