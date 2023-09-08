array=(
    vicuna
    koala
    wizardlm
    sinstruct
    lima
)
for i in "${array[@]}"
do
    echo $i
        python wrap_inference_result.py \
            --dataset_name $i \
            --fname1 xxx1/test_inference \
            --fname2 xxx2/test_inference \
            --save_name xxx1-VS-xxx2
done

