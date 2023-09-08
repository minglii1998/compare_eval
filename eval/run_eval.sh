for eval_file in 'koala_seed_0.json' 'sinstruct_seed_0.json' 'wizardlm_seed_0.json' # 'vicuna_seed_0.json'
do
    python eval.py -qa ./results/${eval_file} -k1 alpaca -k2 alpaca_gpt4 --batch_size 10 --max_tokens 256
    python eval.py -qa ./results/${eval_file} -k1 alpaca_gpt4 -k2 alpaca --batch_size 10 --max_tokens 256
done