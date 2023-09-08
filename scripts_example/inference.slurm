#!/bin/bash

#SBATCH --time=24:00:00             #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                  #Request 1 task
#SBATCH --ntasks-per-node=1         #Request 1 task/core per node
#SBATCH --mem=64gb                 #Request 2560MB (2.5GB) per node
#SBATCH --output=outlogging.%j      #Send stdout/err to "Example1Out.[jobID]"
#SBATCH --account=tianyi-prj-cmsc

# modules needed for running DL jobs. Module restore will also work
source ~/scratch/venv/alp/bin/activate

cd ~/scratch/lima-para

array=(
    logs/xxx1-VS-xxx2/koala_wrap.json
    logs/xxx1-VS-xxx2/lima_wrap.json
    logs/xxx1-VS-xxx2/sinstruct_wrap.json
    logs/xxx1-VS-xxx2/vicuna_wrap.json
    logs/xxx1-VS-xxx2/wizardlm_wrap.json
)
for i in "${array[@]}"
do
    echo $i
        python eval_after_wrap.py \
            --wraped_file $i \
            --batch_size 15 \
            --api_key xxx

done