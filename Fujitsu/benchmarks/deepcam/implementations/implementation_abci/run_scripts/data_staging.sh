#!/bin/sh

if [ $# -lt 5 ]; then
	echo "$0 [data_path] [local_dir] [rank] [size] [local_rank]"
	echo "data staging exit."
	exit 1
fi

data_path=$1
local_dir=$2
rank=$3
nprocs=$4
local_rank=$5
debug=""

# 0: no data staging if data is stored in compute nodes
# 1: remove existing data in compute nodes (if any) and do data staging
force_data_staging=1

if [ ${local_rank} -eq 0 ]; then
    cp ${data_path}/stats.h5 ${local_dir}
fi

start_idx=${rank}
end_idx=2047
idx_step=${nprocs}

train_local_dir="${local_dir}/train/${rank}/"
train_data_prefix="${data_path}/train"

validation_local_dir="${local_dir}/validation/${rank}/"
validation_data_prefix="${data_path}/validation"

# Training
if [ -d ${train_local_dir} ]; then
	if [ ${force_data_staging} -eq 1 ]; then
		rm -rf ${train_local_dir}
	fi
fi

if [ ! -d ${train_local_dir} ]; then
	mkdir -p ${train_local_dir}

	for file_idx in `seq ${start_idx} ${idx_step} ${end_idx}`; do
		train_data="${train_data_prefix}_${file_idx}.tar"

		#echo "rank: $rank is staging: ${train_data} as file_idx: ${file_idx} from [${start_idx}:${idx_step}:${end_idx}]..."
		tar xf ${train_data} -C ${train_local_dir}
	done
fi

# Validation
if [ -d ${validation_local_dir} ]; then
	if [ ${force_data_staging} -eq 1 ]; then
		rm -rf ${validation_local_dir}
	fi
fi

if [ ! -d ${validation_local_dir} ]; then
	mkdir -p ${validation_local_dir}

	for file_idx in `seq ${start_idx} ${idx_step} ${end_idx}`; do
		validation_data="${validation_data_prefix}_${file_idx}.tar"

		#echo "rank: $rank is staging: ${validation_data} as file_idx: ${file_idx} from [${start_idx}:${idx_step}:${end_idx}]..."
		tar xf ${validation_data} -C ${validation_local_dir}
	done
fi

num_local_train_files=`ls ${train_local_dir}/ | wc -l`
num_local_validation_files=`ls ${validation_local_dir}/ | wc -l`
echo "rank: ${rank}, num_local_train_files: ${num_local_train_files}, num_local_validation_files: ${num_local_validation_files}"
