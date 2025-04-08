#!/bin/bash

export CC="mpicc -O2  -g -fpermissive -fPIC"
export CXX="mpic++ -O2  -g -fpermissive -fPIC"

rm -rf build
mkdir -p build
cd build

cmake  ../source
make VERBOSE=1 -j 8
make install 