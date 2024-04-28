CUDA_VISIBLE_DEVICES=0 python OmniFusion/test_grounding_deamon.py \
    --exp_path "./ckpts/Grouning_1B/version_18/" \
    --tokenizer "TinyLlama/TinyLlama-1.1B-Chat-v1.0"\
    --dataset "ref3rec_new_train_500.json"\
    --save_file "deamon_metrics_train500.json"\
    --num_samples 500