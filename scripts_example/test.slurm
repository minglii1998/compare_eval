#!/bin/bash

#SBATCH --time=24:00:00             #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                  #Request 1 task
#SBATCH --ntasks-per-node=1         #Request 1 task/core per node
#SBATCH --mem=64gb                 #Request 2560MB (2.5GB) per node
#SBATCH --output=outlogging.%j      #Send stdout/err to "Example1Out.[jobID]"
#SBATCH --gres=gpu:a100:1           #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu             #Request the GPU partition/queue
#SBATCH --account=tianyi-prj-cmsc

# modules needed for running DL jobs. Module restore will also work
source ~/scratch/venv/alp/bin/activate

cd ~/scratch/lima-para

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
        python generation/generation.py \
            --dataset_name $i \
            --model_name_or_path xxx \
            --max_length 1024 

done