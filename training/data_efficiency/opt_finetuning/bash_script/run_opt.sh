#!/bin/bash


# run_trainer.py \
export CUDA_VISIBLE_DEVICES=0,1
model_name=$1
out_path="./output/${model_name}"
cfg_path="$2"
mkdir -p ${out_path} 
python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port 12345 \
    run_trainer.py \
    --dataset_name ptb_text_only \
    --dataset_config_name penn_treebank \
    --model_name_or_path "facebook/${model_name}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 2 \
    --deepspeed_config ${cfg_path} \
    --deepspeed --seed 1234 --num_warmup_steps 100 \
    --output_dir ${out_path} 
