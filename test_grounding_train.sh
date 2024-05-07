CUDA_VISIBLE_DEVICES=3 python OmniFusion_MSE/test_grounding_deamon.py \
    --exp_path "./ckpts/Grounding_MSE/version_9/" \
    --tokenizer "TinyLlama/TinyLlama-1.1B-Chat-v1.0"\
    --dataset "new_train_500.json"\
    --save_file "deamon_metrics_train500.json"\
    --num_samples 500