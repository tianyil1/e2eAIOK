################## GPT2  ###############
RANDOM_SEED=`date '+%Y-%m-%d-%H-%M-%S'`

#evaluate gpt2
#python run_clm.py \
#    --model_name_or_path gpt2 \
#    --dataset_name wikitext \
#    --dataset_config_name wikitext-2-raw-v1 \
#    --per_device_eval_batch_size 8 \
#    --do_eval \
#    --output_dir /home/vmagent/app/data/LLM/gpt2/evaluate \
#    --overwrite_output_dir \
#    --report_to none

#finetune gpt2
#python run_clm.py \
#    --model_name_or_path gpt2 \
#    --dataset_name wikitext \
#    --dataset_config_name wikitext-2-raw-v1 \
#    --per_device_train_batch_size 8 \
#    --per_device_eval_batch_size 8 \
#    --do_train \
#    --do_eval \
#    --output_dir /home/vmagent/app/data/LLM/gpt2/finetune \
#    --overwrite_output_dir \
#    --report_to none

#finetune gpt2 with no trainer script
#python run_clm_no_trainer.py \
#    --model_name_or_path /home/vmagent/app/data/LLM/gpt2/data \
#    --dataset_name wikitext \
#    --dataset_config_name wikitext-2-raw-v1 \
#    --per_device_train_batch_size 8 \
#    --per_device_eval_batch_size 8 \
#    --output_dir /home/vmagent/app/data/LLM/gpt2/finetune-no-trainer\
#    --report_to none

#evaluate gpt2 with no trainer script
#python run_clm_no_trainer.py \
#    --model_name_or_path /home/vmagent/app/data/LLM/gpt2/finetune \
#    --dataset_name wikitext \
#    --dataset_config_name wikitext-2-raw-v1 \
#    --per_device_eval_batch_size 8 \
#    --output_dir /home/vmagent/app/data/LLM/gpt2/test \
#    --report_to none \
#    --eval

######################### GPT2 with DeNas################
#finetune gpt2 with DeNas
#python run_clm_no_trainer.py \
#    --model_name_or_path /home/vmagent/app/dataset/gpt2/ \
#    --is_denas \
#    --dataset_name wikitext \
#    --dataset_config_name wikitext-2-raw-v1 \
#    --per_device_train_batch_size 8 \
#    --per_device_eval_batch_size 8 \
#    --output_dir /home/vmagent/app/dataset/gpt2/denas-notrainer2\
#    --report_to none 2>&1 | tee GPT2_DENAS_finetune_wikitext_${RANDOM_SEED}.log


######################### GPT2 with DeNas and KD ################
#finetune gpt2 with no trainer with DeNas and KD
python run_clm_no_trainer.py \
    --model_name_or_path /home/vmagent/app/dataset/gpt2 \
    --teacher_model_name_or_path /home/vmagent/app/dataset/gpt2-wikitext2 \
    --is_denas \
    --seed 12345 \
    --is_transferrable \
    --dataset_name wikitext \
    --learning_rate 5e-4 \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --output_dir /home/vmagent/app/dataset/gpt2/denas-kd \
    --report_to none 2>&1 | tee GPT2_DENAS_KD_finetune_wikitext_${RANDOM_SEED}.log

########################GPT-XL################################
#finetune gpt2-xl
#python run_clm.py \
#    --model_name_or_path /home/vmagent/app/data/LLM/gpt2-xl/gpt2-xl \
#    --dataset_name wikitext \
#    --dataset_config_name wikitext-2-raw-v1 \
#    --per_device_train_batch_size 4 \
#    --per_device_eval_batch_size 4 \
#    --do_train \
#    --do_eval \
#    --output_dir /home/vmagent/app/data/LLM/gpt2-xl/finetune \
#    --overwrite_output_dir \
#    --report_to none

########################GPT-J################################
#evaluate gpt-j-6b
#python run_clm.py \
#    --model_name_or_path EleutherAI/gpt-j-6B \
#    --dataset_name wikitext \
#    --dataset_config_name wikitext-2-raw-v1 \
#    --per_device_eval_batch_size 8 \
#    --do_eval \
#    --output_dir /home/vmagent/app/data/LLM/gpt-j-6B/evaluate \
#    --overwrite_output_dir \
#    --report_to none

#finetune gpt-j-6b
#python run_clm.py \
#    --model_name_or_path EleutherAI/gpt-j-6B \
#    --dataset_name wikitext \
#    --dataset_config_name wikitext-2-raw-v1 \
#    --per_device_train_batch_size 8 \
#    --per_device_eval_batch_size 8 \
#    --do_train \
#    --do_eval \
#    --output_dir /home/vmagent/app/data/LLM/gpt-j-6B/finetune \
#    --overwrite_output_dir \
#    --report_to none