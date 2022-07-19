#!/bin/bash

if [ $# -eq 0 ]; then
  echo "$0 [num_nodes] [num_procs_per_node (default: 4)] [mode (default: 1)] [prof: (default: off)] [fraction: 0] [importance: (default: 'disabled')]"
  echo "mode:"
  echo " 0: direct data loading with full dataset,"
  echo " 1: data staging with full dataset (default),"
  echo " 2: data staging with small dataset (for debug),"
  echo " 3: use synthetic data (dummy data) generated in memory (for debug)"
  echo "fraction: the fraction of samples shuffled globally"
  exit 1
fi

num_nodes=$1
let NUM_JOB_NODES=${num_nodes}
if [ $# -gt 1 ]; then
  num_procs_per_node=$2
else
  num_procs_per_node=4
fi
if [ $# -gt 2 ]; then
  mode=$3
else
  mode=1
fi
if [ $# -gt 3 ]; then
  prof=$4
else
  prof=0
fi
if [ $# -gt 4 ]; then
  fraction=$5
else
  fraction=0
fi
if [ $# -gt 5 ]; then
  importance=$6
else
  importance="disabled"
fi


group_id="gcb50300"
# ABCI GC
group_id="gad50726"
group_id="gad50699"
runtime="2:00:00"

log_time=`date +%s`
log_dir=logs
mkdir -p ${log_dir}

TODAY=$(date +%Y%m%d)
branch=`git rev-parse --abbrev-ref HEAD`||exit
SHUFFLING="local_shuffling"
if [ "${fraction}" != "0" ]; then
	SHUFFLING="partial_${fraction}_shuffling"
fi
out_file=${log_dir}/run_training_abci_${TODAY}-${num_nodes}_nodes_${SHUFFLING}_${log_time}.txt

if [ ${mode} -eq 0 ]; then
  echo "direct data load with full dataset" | tee ${out_file}
  data_staging=0
  debug=0
elif [ ${mode} -eq 1 ]; then
  echo "data staging with full dataset" | tee ${out_file}
  data_staging=1
  debug=0
elif [ ${mode} -eq 2 ]; then
  echo "data staging with small dataset (for debug)" | tee ${out_file}
  data_staging=1
  debug=1
elif [ ${mode} -eq 3 ]; then
  echo "use synthetic data (dummy data) generated in memory (for debug)" | tee ${out_file}
  data_staging=0
  debug=2
else
  echo "please specify mode 0, 1, 2, or 3. exit." | tee ${out_file}
  exit 1
fi

qsub -g ${group_id} -l rt_F=${NUM_JOB_NODES} -l h_rt=${runtime} -o ${out_file} -j y -cwd ./run_training_abci.sh \
  ${num_nodes} ${num_procs_per_node} ${data_staging} ${debug} ${prof} ${fraction} ${importance}

sleep 1
