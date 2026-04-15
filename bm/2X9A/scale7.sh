#!/bin/bash
#DSUB -n job_test
#DSUB --job_type cosched
#DSUB -N 16
#DSUB -A root.l6eub2ic
#DSUB -q root.default
#DSUB -R cpu=4;gpu=4
#DSUB -oo scale7_%J.out					
#DSUB -eo scale7_%J.err		

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

mpirun -np 64 -hostfile ${JOB_ID}.nodefile -x PATH -x LD_LIBRARY_PATH -x USE_CUDA -x NUM_THREADS --mca plm_rsh_agent /opt/batch/agent/tools/dstart $dock_home/bin/lightdock -f setup.json -s $STEPS -l 1 

# rm -rf swarm_*
# rm -rf clustered

