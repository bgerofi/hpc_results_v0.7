#!/bin/bash
#SBATCH -J deepcam-cgpu
#SBATCH -C gpu
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 10
#SBATCH --time 30

# Setup software environment
module load cgpu
module load pytorch/v1.6.0-gpu

# Job configuration
rankspernode=8
totalranks=$(( ${SLURM_NNODES} * ${rankspernode} ))
run_tag="deepcam_dryrun_${SLURM_JOB_ID}"
data_dir_prefix="/global/cscratch1/sd/sfarrell/deepcam/data/dry-run"
output_dir=$SCRATCH/deepcam/results/$run_tag

# Create files
mkdir -p ${output_dir}
touch ${output_dir}/train.out

# Run training
srun -u -N ${SLURM_NNODES} -n ${totalranks} --cpu_bind=cores \
     python ../train_hdf5_ddp.py \
     --wireup_method "nccl-slurm" \
     --run_tag ${run_tag} \
     --data_dir_prefix ${data_dir_prefix} \
     --output_dir ${output_dir} \
     --max_inter_threads 2 \
     --model_prefix "classifier" \
     --optimizer "LAMB" \
     --start_lr 4e-3 \
     --lr_schedule type="multistep",milestones="3000 10000",decay_rate="0.1" \
     --lr_warmup_steps 0 \
     --lr_warmup_factor 4 \
     --weight_decay 1e-2 \
     --validation_frequency 50 \
     --training_visualization_frequency 0 \
     --validation_visualization_frequency 0 \
     --logging_frequency 20 \
     --save_frequency 1000 \
     --max_epochs 3 \
     --amp_opt_level O1 \
     --local_batch_size 2 |& tee -a ${output_dir}/train.out
