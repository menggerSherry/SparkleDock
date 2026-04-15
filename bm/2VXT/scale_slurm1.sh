#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --qos=gpugpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --output=./scale1_%j.out			
module load compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/9.8.0.87_cuda12.x  cmake/3.31.6 mpi/4.1.5-gcc11.3.0-cuda12.1-ucx1.14.1


export LD_LIBRARY_PATH=/home/bingxing2/home/scx7753/mxy/SparkleDock/lib:$LD_LIBRARY_PATH
export NUM_THREADS=32
export USE_CUDA=1
STEPS=100
dock_home=/home/bingxing2/home/scx7753/mxy/SparkleDock
rm -rf swarm_*
rm -rf clustered

mpirun -np 1 $dock_home/bin/lightdock -f setup.json -s $STEPS -l 1 


rm -rf swarm_*
rm -rf clustered