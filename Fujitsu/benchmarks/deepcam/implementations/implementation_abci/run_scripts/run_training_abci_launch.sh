#!/bin/bash

if [ $# -lt 11 ]; then
  echo "exit"
  exit
fi

nproc_per_node=$1
data_dir_prefix=$2
stage_dir=$3
output_dir=$4
local_output_dir=$5
num_train_files=$6
num_validation_files=$7
num_data_shards=$8
run_tag=$9
debug=${10}
dummy=${11}
profile=${12}

log_dir=./logs
mkdir -p ${output_dir}

rank=`env | grep OMPI_COMM_WORLD_RANK | awk -F= '{print $2}'`
total_num_procs=`env | grep OMPI_COMM_WORLD_SIZE | awk -F= '{print $2}'`

export OMP_NUM_THREADS=1

if [ ${rank} -eq 0 ]; then
  cp run_training_abci_launch.sh ${log_dir}/run_training_abci_launch_${JOB_ID}.sh
fi

if [ ${stage_dir} != "no" ]; then
  stage_dir="--stage_dir ${stage_dir}"
else
  stage_dir=""
fi

if [ ${debug} -eq 1 ]; then
  debug="--debug"
else
  debug=""
fi

if [ ${dummy} -gt 0 ]; then
  dummy="--dummy"
else
  dummy=""
fi

pin_memory="--pin_memory"

seed=`date +%s`

${profile} python3 ../train_hdf5_ddp.py \
       --wireup_method "mpi" \
       --run_tag ${run_tag} \
       --data_dir_prefix ${data_dir_prefix} \
       --output_dir ${output_dir} \
       --local_output_dir ${local_output_dir} \
       --model_prefix "classifier" \
       --optimizer "LAMB" \
       --start_lr 0.0055 \
       --lr_schedule type="multistep",milestones="800",decay_rate="0.1" \
       --lr_warmup_steps 400 \
       --lr_warmup_factor 1. \
       --weight_decay 1e-2 \
       --validation_frequency 100 \
       --training_visualization_frequency 0 \
       --validation_visualization_frequency 0 \
       --logging_frequency 10 \
       --save_frequency 100 \
       --max_epochs 200 \
       --amp_opt_level O1 \
       --num_global_train_samples ${num_train_files} \
       --num_global_validation_samples ${num_validation_files} \
       --num_train_data_shards ${num_data_shards} \
       --num_validation_data_shards ${num_data_shards} \
       --local_size ${nproc_per_node} \
       --max_inter_threads 4 \
       --shuffle_after_epoch \
       --seed ${seed} \
       --local_batch_size 2 ${stage_dir} ${pin_memory} ${debug} ${dummy}  |& tee -a ${output_dir}/train_${JOB_ID}.out
