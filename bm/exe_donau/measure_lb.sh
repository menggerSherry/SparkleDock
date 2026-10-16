#!/bin/bash
#DSUB -n job_test
#DSUB --job_type cosched
#DSUB -N 128
#DSUB -A root.l6eub2ic
#DSUB -q root.default
#DSUB -R cpu=4;gpu=4
#DSUB -oo scale10_%J.out					
#DSUB -eo scale10_%J.err	
if [ -z $dock_home ]
then
    echo "not found envoriment variable : dock_home"
fi
JOB_ID=${BATCH_JOB_ID}
cat $CCS_ALLOC_FILE | grep ^whshare | awk '{print $1,"slots="$2}' > ${JOB_ID}.nodefile
# ntasks=`cat ${JOB_ID}.nodefile | awk -F '=' '{sum=sum+$2}END{print sum}'`


set -ex
declare -a stringArray
stringArray=("4LW4" "2VXT" "4GAM" "4JCV")
HOME=$dock_home/bm/
STEPS=100
export LD_LIBRARY_PATH=$dock_home/lib:$LD_LIBRARY_PATH
export NUM_THREADS=32
export USE_CUDA=1
for str in "${stringArray[@]}"; do
echo "prepare $str" >> ${HOME}measure.txt

COMPLEX=$str

echo ${HOME}${COMPLEX}
cd ${HOME}${COMPLEX}
# # Setup
rm -rf lightdock* init/ swarm_*

mpirun -np 512 -hostfile ${JOB_ID}.nodefile -x PATH -x LD_LIBRARY_PATH -x USE_CUDA -x NUM_THREADS --mca plm_rsh_agent /opt/batch/agent/tools/dstart $dock_home/bin/sparkledock_lb -f setup.json -s $STEPS -l 1 >> ${HOME}measure.txt