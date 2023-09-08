MODEL_NAME='alpaca-t45-3k'
DATA_DIR=/group-volume/lichang/LIMA-alpaca/training_data/alpaca_t45_3k.json
OUTPUT_DIR=/group-volume/lichang/${MODEL_NAME}
GENERATION_DIR=/group-volume/lichang/bad_alpaca/generation

cd ${GENERATION_DIR}
export CUDA_VISIBLE_DEVICES=0
for training_data_source_name in ${MODEL_NAME}
# 'alpaca_gpt4'
do
    model_name=${training_data_source_name}

    for dataset_name in 'vicuna' 'koala' 'wizardlm' 'sinstruct'
    do
        python generation.py \
            --model_name_or_path ${OUTPUT_DIR} \
            --training_data_source_name ${training_data_source_name} \
            --dataset_name ${dataset_name}
        # echo ‘--------’
        # echo ${training_data_source_name}
        # echo ${model_name}
        # echo ${dataset_name}
    done
done