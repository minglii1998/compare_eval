python -m truthfulqa.evaluate \
    --models chavinlo/alpaca-native \
    --metrics mc \
    --input_path TruthfulQA.csv \
    --output_path TruthfulQA_answers.csv \
    --cache_dir /data/bobchen/ \
    --device 3