#!/bin/bash
set -xe

function main {
    source oob-common/common.sh
    # set common info
    init_params $@
    fetch_device_info
    set_environment

    # requirements
    pip install evaluate datasets==2.8.0
    pip uninstall tokenizers transformers -y
    python setup.py develop
    export WANDB_DISABLED=True

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        #
        for batch_size in ${batch_size_list[@]}
        do
            # clean workspace
            logs_path_clean

	    # get batchSize
    	    if [[ ${mode_name} == "train" ]];then
    	        perf_mode=" --do_train "
	    	perf_mode+=" --per_device_train_batch_size ${batch_size} "
    	    else # realtime
    	        perf_mode=" --do_eval "
	    	perf_mode+=" --per_device_eval_batch_size ${batch_size} "
    	    fi

	    # download gpt2 model
            #if [ "${model_name}" == "token-classification+gpt2" ];then
            #    rm -rf ./gpt2-model-for-classification_2/
            #    python ${workload_dir}/gpt2-model-for-classification.py --precision ${precision}
            #fi

            # generate launch script for multiple instance
            if [ "${OOB_USE_LAUNCHER}" == "1" ] && [ "${device}" == "cpu" ];then
                generate_core_launcher
            else
                generate_core
            fi
            # launch
            echo -e "\n\n\n\n Running..."
            source ${excute_cmd_file}
            echo -e "Finished.\n\n\n\n"
            # collect launch result
            collect_perf_logs
        done
    done
}

# run
function generate_core {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${device_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        # instances
        if [ "${device}" == "cpu" ];then
            OOB_EXEC_HEADER=" numactl -m $(echo ${device_array[i]} |awk -F ';' '{print $2}') "
            OOB_EXEC_HEADER+=" -C $(echo ${device_array[i]} |awk -F ';' '{print $1}') "
        elif [ "${device}" == "cuda" ];then
            OOB_EXEC_HEADER=" CUDA_VISIBLE_DEVICES=${device_array[i]} "
	    if [[ "${mode_name}" == "realtime" ]];then
	        addtion_options+=" --nv_fuser "
	    fi
	fi
	# remove jit, longformers fail with "ValueError: not enough values to unpack (expected 2, got 1)"
        printf " ${OOB_EXEC_HEADER} \
	    python examples/${framework}/$(echo ${EXAMPLE_ARGS}) \
		${perf_mode} --no_cuda --overwrite_output_dir --output_dir /tmp/tmp0 \
	        --num_iters $num_iter --num_warmup $num_warmup \
		--channels_last $channels_last --precision $precision \
		--device ${device} \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
        if [ "${numa_nodes_use}" == "0" ];then
            break
        fi
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

function generate_core_launcher {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${device_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        printf "python -m launch --enable_jemalloc \
                    --core_list $(echo ${device_array[@]} |sed 's/;.//g') \
                    --log_file_prefix rcpi${real_cores_per_instance} \
                    --log_path ${log_dir} \
                    --ninstances ${#device_array[@]} \
                    --ncore_per_instance ${real_cores_per_instance} \
            tools/infer.py --weights $CKPT_DIR \
                --source $DATASET_DIR \
                --num_iter $num_iter --num_warmup $num_warmup \
                --channels_last $channels_last --precision $precision \
                ${addtion_options} \
        > /dev/null 2>&1 &  \n" |tee -a ${excute_cmd_file}
        break
    done
    echo -e "\n wait" >> ${excute_cmd_file}
    # download launcher
    wget --no-proxy -O launch.py http://mengfeil-ubuntu.sh.intel.com/share/launch.py
}

# download common files
rm -rf oob-common && git clone https://github.com/intel-sandbox/oob-common.git -b gpu_oob

# Start
main "$@"
