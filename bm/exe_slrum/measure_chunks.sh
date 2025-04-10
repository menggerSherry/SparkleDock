set -ex
declare -a stringArray
stringArray=("3LVK")
HOME=/share/home/mengxiangyu/sparkledock/bm/
STEPS=100
dock_home=/share/home/mengxiangyu/sparkledock
export CHUNKS=2
# rm -rf ${HOME}measure.txt
export LD_LIBRARY_PATH=/share/home/mengxiangyu/sparkledock/lib:$LD_LIBRARY_PATH
export NUM_THREADS=20
export USE_CUDA=1


for str in "${stringArray[@]}"; do

echo "prepare $str" >> ${HOME}measure.txt

COMPLEX=$str

echo ${HOME}${COMPLEX}
cd ${HOME}${COMPLEX}
# # Setup
rm -rf lightdock* init/ swarm_*
# cp ../${COMPLEX}_A_noh.pdb ../${COMPLEX}_B_noh.pdb .

# rm -rf swarm_*
# rm -rf clustered
pwd
start_time=$(date +"%s")
ncu --set roofline -f -o pipeopt_${COMPLEX}  $dock_home/bin/lightdock -f setup.json -s $STEPS -l 1 
# $dock_home/bin/lightdock -f setup.json -s $STEPS -l 1 

end_time=$(date +"%s")
execution_time=$((end_time - start_time))
rm -rf lightdock*  init/ swarm_*
echo "Total Execution Time: $execution_time seconds" >> ${HOME}measure.txt

done