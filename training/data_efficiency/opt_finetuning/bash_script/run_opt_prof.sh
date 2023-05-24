#!/bin/bash

log_dir=./ds_train_logs/
gpu_logs=${log_dir}/gpu_logs
cpu_logs=${log_dir}/cpu_logs
mkdir -p ${gpu_logs}
mkdir -p ${cpu_logs}


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_num=$3
model_name=$1
out_path="./output/${model_name}"
cfg_path="$2"
policy=$4
bs=1
mkdir -p ${out_path} 
base_f=${model_name}_${gpu_num}_${policy}_${bs}
gpu_log="${gpu_logs}/${base_f}.qdrep"
cpu_log="${gpu_logs}/${base_f}.qdrep"
nsys profile  -c cudaProfilerApi -f true --stats true  -o ${gpu_log} \
    python -m torch.distributed.launch --nproc_per_node=${gpu_num} \
    --master_port 12345 \
    run_trainer_prof.py \
    --dataset_name ptb_text_only \
    --dataset_config_name penn_treebank \
    --model_name_or_path "facebook/${model_name}" \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 2 \
    --deepspeed_config ${cfg_path} \
    --deepspeed --seed 1234 --num_warmup_steps 100 \
    --output_dir ${out_path} \
    --cpu_log_path ${cpu_log}
