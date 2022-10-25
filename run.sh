pip install evaluate
python setup.py develop


python ./examples/pytorch/text-classification/run_glue.py --task_name MRPC --model_name_or_path allenai/longformer-base-4096  --do_eval --no_cuda --overwrite_output_dir --output_dir /tmp/tmp0 --per_device_eval_batch_size 1 --num_iters 200 --num_warmup 20 --channels_last 1 --precision float32  --device cpu
