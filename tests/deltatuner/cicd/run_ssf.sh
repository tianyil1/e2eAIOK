set -x

cd /home/vmagent/app/e2eaiok/e2eAIOK/deltatuner
pip install -e .
pip uninstall wandb -y

cd /home/vmagent/app/e2eaiok

mkdir -p log

DATA_PATH="/home/vmagent/app/data"

#run mpt with ssf, bf16
python example/instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "$DATA_PATH/mpt-7b" \
        --train_file "$DATA_PATH/alpaca_data.json" \
        --dataset_concatenation \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --validation_split_percentage 30 \
        --do_train \
        --do_eval \
        --learning_rate 1e-4 \
        --num_train_epochs 1 \
        --logging_steps 100 \
        --save_total_limit 1 \
        --log_level info \
        --save_strategy epoch \
        --trust_remote_code True \
        --no_cuda \
        --output_dir "$DATA_PATH/mpt-7b-ssf-allmodules-bf16" \
        --delta ssf \
        --denas False \
        --debugs \
        2>&1 | tee log/mpt-ssf-run-allmodules-bf16-1epoch.log
    
#run llama2 with ssf, bf16
python example/instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path "$DATA_PATH/Llama-2-7b-hf" \
        --train_file "$DATA_PATH/alpaca_data.json" \
        --dataset_concatenation \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --validation_split_percentage 30 \
        --do_train \
        --do_eval \
        --learning_rate 1e-4 \
        --num_train_epochs 1 \
        --logging_steps 100 \
        --save_total_limit 1 \
        --log_level info \
        --save_strategy epoch \
        --trust_remote_code True \
        --no_cuda \
        --output_dir "$DATA_PATH/llama2-7b-ssf-bf16" \
        --delta ssf \
        --denas False \
        --debugs \
        2>&1 | tee log/llama2-7b-ssf-bf16-1epoch.log