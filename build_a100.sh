#!/bin/bash

# salloc -N 1 --gres=gpu:1 --qos=gpugpu -x paraai-n32-h-01-agent-[1,4,7-8,16-17,25,27-31]
module load compilers/cuda/11.8 compilers/gcc/11.3.0   cmake/3.31.6 mpi/openmpi4.1.5-gcc11.3.0-cuda11.8-ucx1.12.1

export CC="mpicc -O2  -g -fpermissive -fPIC"
export CXX="mpic++ -O2  -g -fpermissive -fPIC"

rm -rf build bin lib
mkdir -p build
cd build

cmake  ../source
make VERBOSE=1 -j 8
make install 