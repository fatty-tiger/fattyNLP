CUDA_VISIBLE_DEVICES='4' \
/data/miniconda3/envs/deepnlp/bin/python train.py \
    --model_name_or_path /data/pretrained_models/ernie-3.0-nano-zh \
    --entity_num 10 \
    --inner_dim 64 \
    --rope \
    --output_dir 'output/v1' \
    --overwrite_output_dir \
    --train_data /data/jiangjie/fattyNLP/dataset/ner/cluener/train.jsonl \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --logging_steps 50 \
    --save_steps 20000

# torchrun --nproc_per_node 1 train.py \
#     --model_name_or_path /data/pretrained_models/ernie-3.0-nano-zh \
#     --entity_num 10 \
#     --inner_dim 16 \
#     --rope \
#     --output_dir 'output/v1' \
#     --overwrite_output_dir \
#     --train_data /data/jiangjie/fattyNLP/dataset/ner/cluener/train.jsonl \
#     --learning_rate 5e-5 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 64 \
#     --dataloader_drop_last True \
#     --fp16 \
#     --warmup_ratio 0.1 \
#     --logging_steps 50 \
#     --save_steps 20000
