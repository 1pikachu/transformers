#!/bin/bash
set -xe

function main {
    # set common info
    source common.sh
    init_params $@
    fetch_device_info
    set_environment

    # requirements
    pip install evaluate datasets
    pip uninstall transformers -y
    pip install -e .

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
            generate_core
            # launch
            echo -e "\n\n\n\n Running..."
            #cat ${excute_cmd_file} |column -t > ${excute_cmd_file}.tmp
            #mv ${excute_cmd_file}.tmp ${excute_cmd_file}
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
        if [ "${device}" != "cuda" ];then
            OOB_EXEC_HEADER=" numactl -m $(echo ${device_array[i]} |awk -F ';' '{print $2}') "
            OOB_EXEC_HEADER+=" -C $(echo ${device_array[i]} |awk -F ';' '{print $1}') "
        else
            OOB_EXEC_HEADER=" CUDA_VISIBLE_DEVICES=${device_array[i]} "
        fi
        printf " ${OOB_EXEC_HEADER} \
	    python examples/tensorflow/language-modeling/run_mlm.py \
	    	--model_name_or_path bert-base-multilingual-uncased \
		--dataset_name wikitext  --dataset_config_name wikitext-103-raw-v1 \
		--output_dir /tmp/tmp0 \
		--epochs 3 --num_iter ${num_iter} --num_warmup 1 \
		--precision ${precision} --per_device_eval_batch_size $batch_size \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

# download common files
wget -q -O common.sh https://raw.githubusercontent.com/mengfei25/oob-common/gpuoob/common.sh

# Start
main "$@"
