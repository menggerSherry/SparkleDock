#!/bin/bash
#SBATCH -N 8
#SBATCH --gres=gpu:4
#SBATCH --qos=gpugpu
#SBATCH -c 32
#SBATCH --ntasks-per-node=4
#SBATCH -n 32
#SBATCH --output=./scale6_%j.out	

JOB_ID=${BATCH_JOB_ID}
cat $CCS_ALLOC_FILE | grep ^whshare | awk '{print $1,"slots="$2}' > ${JOB_ID}.nodefile
# ntasks=`cat ${JOB_ID}.nodefile | awk -F '=' '{sum=sum+$2}END{print sum}'`


export LD_LIBRARY_PATH=/home/l6eub2ic/whcs-share31/mengxiangyu/SparkleDock/lib:$LD_LIBRARY_PATH
export NUM_THREADS=32
export USE_CUDA=1
STEPS=100
dock_home=/home/l6eub2ic/whcs-share31/mengxiangyu/SparkleDock
rm -rf swarm_*
rm -rf clustered

mpirun -np 32 -hostfile ${JOB_ID}.nodefile -x PATH -x LD_LIBRARY_PATH -x USE_CUDA -x NUM_THREADS --mca plm_rsh_agent /opt/batch/agent/tools/dstart $dock_home/bin/lightdock -f setup.json -s $STEPS -l 1 

# rm -rf swarm_*
# rm -rf clustered

