#!/bin/bash
set -x
mkdir -p log models

# fine-tune mpt-7b with lora
python example/instruction_tuning_pipeline/finetune_clm.py \
    --model_name_or_path "/home/vmagent/app/dataset/mpt-7b" \
    --train_file "/home/vmagent/app/dataset/stanford_alpaca/alpaca_data.json" \
    --dataset_concatenation \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --validation_split_percentage 30 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --logging_steps 100 \
    --save_total_limit 1 \
    --log_level info \
    --save_strategy epoch \
    --output_dir models/mpt_lora_model \
    --peft lora \
    --denas False \
    --trust_remote_code True \
    --no_cuda \
    --bf16 True 2>&1 | tee log/mpt-lora-run-1epoch.log

# fine-tune llama2-7b with lora
python example/instruction_tuning_pipeline/finetune_clm.py \
    --model_name_or_path "/home/vmagent/app/dataset/Llama-2-7b-hf" \
    --train_file "/home/vmagent/app/dataset/stanford_alpaca/alpaca_data.json" \
    --dataset_concatenation \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --validation_split_percentage 30 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --logging_steps 100 \
    --save_total_limit 1 \
    --log_level info \
    --save_strategy epoch \
    --output_dir models/llama2_lora_model \
    --peft lora \
    --denas False \
    --trust_remote_code True \
    --no_cuda \
    --bf16 True 2>&1 | tee log/llama2-lora-run-1epoch.log


# used for assign target modules to lora
# --lora_target_modules q_proj v_proj k_proj o_proj up_proj down_proj \
# --lora_target_modules Wqkv out_proj up_proj down_proj \