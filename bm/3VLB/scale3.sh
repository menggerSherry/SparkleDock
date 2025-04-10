#!/bin/bash
#SBATCH --gpus=4
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 32
#SBATCH --output=./scale3_%j.out

# export NCCL_ALGO=Ring
# export NCCL_MAX_NCHANNELS=16
# export NCCL_MIN_NCHANNELS=16
# export NCCL_DEBUG=INFO
# export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
# export NCCL_IB_HCA=mlx5_0,mlx5_2
# export NCCL_IB_GID_INDEX=3
# export OMPI_MCA_btl_openib_warn_no_device_params_found=0
# export OMPI_MCA_btl_openib_verbose=0
# export OMPI_MCA_orte_base_help_aggregate=0
# export OMPI_MCA_btl_openib_warn_no_cpcs_for_port=0

export LD_LIBRARY_PATH=/home/bingxing2/home/scx8jwu/mxy/SparkleDock/lib:$LD_LIBRARY_PATH
export NUM_THREADS=32
export USE_CUDA=1
STEPS=100
dock_home=/home/bingxing2/home/scx8jwu/mxy/SparkleDock
rm -rf swarm_*
rm -rf clustered

srun $dock_home/bin/lightdock -f setup.json -s $STEPS -l 1 

rm -rf swarm_*
rm -rf clustered