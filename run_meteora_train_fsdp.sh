export MOELINEAR_USE_ACCELERATE_FWD=0

accelerate launch --num_processes=4 --config_file "configs/fsdp_config.yaml" meteora_train.py \
--model_name "" \
--tasks_datasets_prefix "" \
--lora_path_prefix "" \
--default_task "alpaca" \
--max_seq_len 4096 \
--max_steps 5000 \
--logging_steps 1 \
--eval_steps 500 \
--save_steps 500 \
--bf16 True \
--packing True \
--output_dir "" \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2 \
--per_device_eval_batch_size 8 \
--dataset_text_field "text" \
--learning_rate 7e-4  \
--lr_scheduler_type "cosine" \
--weight_decay 0.01 \
--warmup_ratio 0.05 \
--use_flash_attn True \
--use_gradient_checkpointing       