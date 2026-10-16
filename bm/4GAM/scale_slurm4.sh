#!/bin/bash
#SBATCH -N 2
#SBATCH --gres=gpu:4
#SBATCH --qos=gpugpu
#SBATCH -c 32
#SBATCH --ntasks-per-node=4
#SBATCH -n 8
#SBATCH --output=./scale4_%j.out

# JOB_ID=${BATCH_JOB_ID}
# cat $CCS_ALLOC_FILE | grep ^whshare | awk '{print $1,"slots="$2}' > ${JOB_ID}.nodefile
# ntasks=`cat ${JOB_ID}.nodefile | awk -F '=' '{sum=sum+$2}END{print sum}'`
# JOB_ID=${BATCH_JOB_ID}
# cat $CCS_ALLOC_FILE | grep ^whshare | awk '{print $1,"slots="$2}' > ${JOB_ID}.nodefile
module load compilers/cuda/11.8 compilers/gcc/11.3.0   cmake/3.31.6 mpi/openmpi4.1.5-gcc11.3.0-cuda11.8-ucx1.12.1

export LD_LIBRARY_PATH=/home/bingxing2/home/scx7753/mxy/SparkleDock/lib:$LD_LIBRARY_PATH
export NUM_THREADS=32
export USE_CUDA=1
STEPS=100
dock_home=/home/bingxing2/home/scx7753/mxy/SparkleDock
rm -rf swarm_*
rm -rf clustered

mpirun -np 8  $dock_home/bin/sparkledock -f setup.json -s $STEPS -l 1 

# rm -rf swarm_*
# rm -rf clustered

