It's fine to directly use Alpaca envriment.

1. 
scripts_example/test.slurm
This code will generate a new directory named "test_inference" in the model path.

2. 
scripts_example/wrap.sh
This process do not need to use sbatch.
fname1: the model path for the 1st model
fname2: the model path for the 2nd model
save_name: save name in logs directory

3. 
scripts_example/inference.slurm
Use chatGPT or GPT4 for the evalution. 
CAN NOT BE RUN ON ZARATAN.
Remeber to modify the array, as the previous save_name.

4.
review_score.py
If want to draw the comparing figure.