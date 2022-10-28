pip install evaluate datasets
pip install -e .


python examples/tensorflow/language-modeling/run_mlm.py --model_name_or_path bert-base-multilingual-uncased --output_dir /tmp/tmp0 --dataset_name wikitext  --dataset_config_name wikitext-103-raw-v1 --epochs 3 --num_iter 100 --num_warmup 1 --precision float16 --per_gpu_eval_batch_size 1
