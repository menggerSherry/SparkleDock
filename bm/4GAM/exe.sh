#!/bin/bash
#SBATCH --job-name=mxy
#SBATCH --partition=a800
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -w g07
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --output=./log/log_process%j.out
#SBATCH --error=./log/log_process%j.err


export LD_LIBRARY_PATH=/home/l6eub2ic/whcs-share31/mengxiangyu/sparkledock/lib:$LD_LIBRARY_PATH
export NUM_THREADS=8
export USE_CUDA=1
STEPS=2
dock_home=/home/l6eub2ic/whcs-share31/mengxiangyu/sparkledock

rm -rf swarm_*
rm -rf clustered
# valgrind --log-file=./valgrind_report.log --show-leak-kinds=all --show-possibly-lost=no --tool=memcheck --leak-check=full ./bin/lightdock -f setup.json -s 100 -l 1 > output2.txt
start_time=$(date +"%s")


# ncu --set roofline -f -o opt10  $dock_home/bin/lightdock -f setup.json -s $STEPS -l 1 
# nsys profile --stats=true --sample=cpu --trace=openmp $dock_home/bin/lightdock -f setup.json -s $STEPS -l 1 
mpirun -np 1 $dock_home/bin/lightdock -f setup.json -s $STEPS -l 1 
# perf record  -e cycles:u -j any,u -a -o perf.data $dock_home/bin/lightdock -f setup.json -s $STEPS -l 1 
end_time=$(date +"%s")
execution_time=$((end_time - start_time))
echo "Total Execution Time: $execution_time seconds"
# lgd_cluster_bsas.py gso_100.out

# cd swarm_0
# python ../analysis/cluster.py gso_100.out