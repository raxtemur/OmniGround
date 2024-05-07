CUDA_VISIBLE_DEVICES=3 python OmniFusion_MSE/test_grounding_deamon.py \
    --exp_path "./ckpts/Grounding_MSE/version_10/" \
    --tokenizer "TinyLlama/TinyLlama-1.1B-Chat-v1.0"\
    --save_file "deamon_metrics.json"