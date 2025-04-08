#!/bin/bash
#SBATCH -N 2
#SBATCH --gres=gpu:4
#SBATCH --qos=gpugpu
#SBATCH -c 32
#SBATCH --ntasks-per-node=4
#SBATCH -n 8
#SBATCH --output=./scale4_%j.out

# export NCCL_ALGO=Ring
# export NCCL_MAX_NCHANNELS=16
# export NCCL_MIN_NCHANNELS=16
# export NCCL_DEBUG=INFO
# export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
# export NCCL_IB_HCA=mlx5_0,mlx5_2
# export NCCL_IB_GID_INDEX=3

# srun bash -c "while true; do echo \$(hostname): \$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits) >> gpu_usage_\$(hostname).log; sleep 5; done" &

export LD_LIBRARY_PATH=/home/bingxing2/home/scx8jwu/mxy/sparkledock/lib:$LD_LIBRARY_PATH
export NUM_THREADS=32
export USE_CUDA=1
STEPS=100
dock_home=/home/bingxing2/home/scx8jwu/mxy/sparkledock
rm -rf swarm_*
rm -rf clustered

srun $dock_home/bin/lightdock -f setup.json -s $STEPS -l 1 

rm -rf swarm_*
rm -rf clustered

