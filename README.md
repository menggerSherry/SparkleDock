# SparkleDock: Scalable High-Fidelity Macromolecular Docking for GPU-Accelerated Supercomputers

# Overview

SparkleDock is a scalable Glowworm Swarm Optimization (GSO) docking system on GPU supercomputers, enabling large-scale flexible macromolecule docking. It supports:

*  A fine-grained, agent-level parallelization to match the SM in GPU.
*  TensorCore accelerated CUDA kernel design to enhance the computing efficiency.
*  Efficient MPI design to adaptively scale across multi-GPU systems.

SparkleDock can accelerate the macromolecule docking by over two orders of magnitude compared to the existing flexible macromolecule docking.

# Install SparkleDock

## Requirements

### software requirements
```
openmpi/4.1.5 and over
cuda/11.8.0 and over 
gcc/11.3.0 and over
cmake/3.16.5 and over
```
### Hardware requirements

NVIDIA GPU or multi-GPU systems.

## Install SparkleDock

```shell
export CC="mpicc -O2  -g -fpermissive -fPIC"
export CXX="mpic++ -O2  -g -fpermissive -fPIC"

rm -rf build
mkdir -p build
cd build

# if the GPU do not support FP64 TensorCores, add -DBUILD_4090=ON
cmake  ../source
make VERBOSE=1 -j 8
make install 
```
For your convience, we provide the install shell scripts for 4090 and A100 GPUs(i.e., `build_4090.sh` and `build_a100.sh`)

# Benchmark dataset 

The table below shows the benchmark dataset from The Protein-Protein Benchmark 5 and Affinity Benchmark Version 2 (BM5). The Category denotes the marcromolefcule type: antibody-antigen (A); enzyme–inhibitor (EI); enzyme–substrate (ES); enzyme complex with a regulatory or accessory chain (ER); others, G-protein containing (OG); others, receptor containing (OR); others, miscellaneous (OX). The Swarms denotes the calculated swarm numbner by GSO. The Rec Atoms and Lig Atoms denotes the backbone atom number. 

| **Complex** | **Category** | **Swarms** | **Rec Atoms** | **Lig Atoms** |
|-------------|--------------|-------------|----------------|----------------|
| **2VXT**    | A            | 402         | 3002           | 1274           |
| **3VLB**    | EI           | 336         | 3052           | 1658           |
| **2A1A**    | ES           | 271         | 2039           | 1440           |
| **2GTP**    | OG           | 298         | 2516           | 1061           |
| **2X9A**    | OR           | 130         | 757            | 481            |
| **1RKE**    | OX           | 294         | 2030           | 1256           |
| **3LVK**    | ER           | 620         | 6143           | 633            |
| **4GAM**    | ER           | 1245        | 17307          | 1125           |
| **4JCV**    | OX           | 695         | 6032           | 1766           |
| **4LW4**    | ES           | 558         | 6058           | 1138           |


# Execute the SparkleDock


## Single GPU execution 
Here is the example of the docking on single GPU.
```shell
cd *SparkleDock root path*
export dock_home=`pwd`
# library path
export LD_LIBRARY_PATH=$dock_home/lib:$LD_LIBRARY_PATH
# simulation steps
STEPS=100
# docking dataset
Complex=2VXT
# use cuda, developing
USE_CUDA=1

cd ${HOME}${Complex}
# remove previous calculations
rm -rf lightdock* init/ swarm_*
# execute calculation
mpirun -np 1 $dock_home/bin/lightdock -f setup.json -s $STEPS -l 1 
```

## multi-GPU execution
Here is the example of the SparkleDock on multi-GPUs

```shell
cd *SparkleDock root path*
export dock_home=`pwd`
# library path
export LD_LIBRARY_PATH=$dock_home/lib:$LD_LIBRARY_PATH
# simulation steps
STEPS=100
# docking dataset
Complex=2VXT
# number of GPUs (i.e. number of the MPI ranks)
GPUS=4

USE_CUDA=1

cd ${HOME}${Complex}
# remove previous calculations
rm -rf lightdock* init/ swarm_*
# execute calculation
mpirun -np $GPUS $dock_home/bin/lightdock -f setup.json -s $STEPS -l 1 
```

if you wish to execute the SparkleDock on multi-GPU supercomputers, we recommend to use the job scheduling systems (e.g., `slurm`, `LSF`, etc). The example of the `slurm` script is denoted as follows:

```shell
#!/bin/bash
#SBATCH -N *nodes*
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH -n *gpus*

export LD_LIBRARY_PATH=$dock_home/lib:$LD_LIBRARY_PATH
# simulation steps
STEPS=100
# docking dataset
Complex=2VXT
# number of GPUs (i.e. number of the MPI ranks)
GPUS=4

USE_CUDA=1

# remove previous calculations
rm -rf lightdock* init/ swarm_*
# execute calculation
mpirun -np $GPUS $dock_home/bin/lightdock -f setup.json -s $STEPS -l 1 

```

# Citation

```
@misc{meng2026scalablehighfidelitymacromoleculardocking,
      title={Scalable High-Fidelity Macromolecular Docking for GPU-Accelerated Supercomputers}, 
      author={Xiangyu Meng and Peng Chen and Mingzhen Li and Jianmin Wang and Sen Wang and Guangming Tan and Weile Jia and Mohamed Wahib and Tao Luo and Xun Wang},
      year={2026},
      eprint={2608.07078},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2608.07078}, 
}
```
