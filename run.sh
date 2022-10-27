pip install evaluate datasets
pip install -e .


python examples/tensorflow/language-modeling/run_mlm.py --model_name_or_path bert-base-multilingual-uncased --output_dir /tmp/tmp0 --dataset_name wikitext  --dataset_config_name wikitext-103-raw-v1 --epochs 1 --num_iter 10 --num_warmup 0 --precision float16
