#!/bin/bash -i

if [ $# -lt 1 ]; then
  echo "$0 [num_nodes] [num_procs_per_node (default: 4)] [data_staging (default: on)] [load_debug_data (default: off)] [prof: (default: off)] [fraction: (default 0)]"
  echo "load_debug_data: {0: no, 1:yes (from files), 2: yes (dummy data)}"
  exit 1
fi

#. ~/miniconda3/etc/profile.d/conda.sh
#conda activate py37-pytorch
#
#. /etc/profile.d/modules.sh
#
#module load gcc/7.4.0
#module load openmpi/2.1.6
#module load cuda/10.2/10.2.89
#module load cudnn/7.6/7.6.5
#module load nccl/2.7/2.7.8-1
#
#export CUDA_HOME=/apps/cuda/10.2.89
#export CUDNN_LIB_DIR=/apps/cudnn/7.6.5/cuda10.2
#export CUDNN_INCLUDE_DIR=$CUDNN_LIB_DIR/include
#export CUDNN_LIBRARY=$CUDNN_LIB_DIR/lib64


source /etc/profile.d/modules.sh
module purge
module load gcc/9.3.0 python/3.8/3.8.7 openmpi/4.0.5 cuda/11.1/11.1.1 cudnn/8.0/8.0.5 nccl/2.8/2.8.4-1
source ~/venv/python-3.8.7+pytorch-1.8.1+horovod-0.22.0/bin/activate

echo "job_id: ${JOB_ID}"

now=`date +%Y%m%d`
run_tag="${now}-JOB_${JOB_ID}"

#directories
data_path_original=/bb/mlperfhpc/deepcam/original/All-Hist/
#data_path_reformatted=/bb/mlperfhpc/deepcam/reformatted/
#data_path_reformatted=/groups/gcb50300/DeepCAM/128GB/reformatted/
data_path_reformatted=/groups/gcb50300/DeepCAM/full/
local_dir=${SGE_LOCALDIR}

fs_output_dir="./runs/${run_tag}"
local_output_dir="${local_dir}/${run_tag}"
output_dir=${fs_output_dir}
log_dir=./logs

#prepare dir:
mkdir -p ${fs_output_dir}
mkdir -p ${log_dir}

hostfile=${log_dir}/hostfile_${JOB_ID}

if [ $# -gt 1 ]; then
  nproc_per_node=$2
else
  nproc_per_node=4
fi

num_nodes=$1
run_tag="${run_tag}-${num_nodes}_nodes"


cp $SGE_JOB_HOSTLIST ${hostfile}

total_num_procs=$((${num_nodes} * ${nproc_per_node}))
echo "total_num_procs: ${total_num_procs}, num_nodes: ${num_nodes}, num_procs_per_node: ${nproc_per_node}"

if [ ${total_num_procs} -lt 1 ]; then
  echo "total num processes < 1. exit." 
  exit
fi

if [ $# -gt 2 ]; then
  data_staging=$3
else
  data_staging=1
fi

if [ $# -gt 3 ]; then
  debug=$4
else
  debug=0
fi

if [ $# -gt 4 ]; then
  prof=$5
else
  prof=0
fi

if [ $# -gt 5 ]; then
  fraction=$6
else
  fraction=0
fi

if [ ${fraction} == "0" ]; then
	run_tag="${run_tag}-local_shuffling"
else
	run_tag="${run_tag}-partial_${fraction}_shuffling"
fi
echo ${run_tag}

if [ ${debug} -eq 2 ]; then
  dummy=1
else
  dummy=0
fi

if [ ${debug} -eq 1 ] && [ ${data_staging} -le 0 ]; then
  echo "combination of debug mode 1 and direct data loading from file system is not supported."
  exit
fi

if [ ${dummy} -eq 1 ] && [ ${data_staging} -gt 0 ]; then
  echo "dummy and data_staging combination is not supported."
  exit 
fi


#mpi options
mpioptions="-mca pml ob1 -map-by ppr:${nproc_per_node}:node"

if [ ${prof} -gt 0 ]; then
  echo "prof"
  profile="nvprof -o profile_${JOB_ID}.nvp"
else
  profile=""
fi

num_train_files=`ls ${data_path_original}/train | grep data | wc -l`
num_validation_files=`ls ${data_path_original}/validation | grep data | wc -l`

if [ ${data_staging} -gt 0 ]; then
  data_dir_prefix=${data_path_reformatted}
  stage_dir=${local_dir}
  num_data_shards=${nproc_per_node}
else
  data_dir_prefix=${data_path_original}
  stage_dir="no"
  num_data_shards=${total_num_procs}
fi

#grep -v g0621 ${PE_HOSTFILE} | grep -v g0905 | cut -d" " -f 1 > ~/tmp/pe_hostfile
if ! ${HOME}/bin/check_gpus.sh ${num_nodes}; then
	echo "error: faulty GPUs?"
	exit 1
fi

#run the stuff
mpirun -hostfile ~/tmp/${JOB_ID}_hostfile -n ${total_num_procs} ${mpioptions} ./run_training_abci_launch.sh \
  ${nproc_per_node} \
  ${data_dir_prefix} \
  ${stage_dir} \
  ${fs_output_dir} \
  ${local_output_dir} \
  ${num_train_files} \
  ${num_validation_files} \
  ${num_data_shards} \
  ${run_tag} \
  ${debug} \
  ${dummy} \
  ${fraction} \
  ${profile}

rm ~/tmp/${JOB_ID}_hostfile

echo "copy output directory"
cp -r ${local_output_dir}/* ${fs_output_dir}

echo "done."
