# python generation.py --dataset_name koala --model_name_or_path /group-volume/lichang/alpaca

cd /home/user/stanford_alpaca
export CUDA_VISIBLE_DEVICES=0
for training_data_source_name in 'alpaca' 
# 'alpaca_gpt4'
do
    model_name=${training_data_source_name}
    mkdir -p /group-volume/lichang/${model_name}/${model_name}

    for dataset_name in 'koala' 'vicuna' 'koala'
    do
        python generation.py \
            --model_name_or_path /group-volume/lichang/${model_name} \
            --training_data_source_name ${training_data_source_name} \
            --dataset_name ${dataset_name}
        mv /group-volume/lichang/${model_name}/${dataset_name} /group-volume/lichang/${model_name}/${model_name} 
        # echo ‘--------’
        # echo ${training_data_source_name}
        # echo ${model_name}
        # echo ${dataset_name}
    done
done