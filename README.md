# LIMA-paraphrase
LIMA paraphrase project. When the models are ready(training finished), inference firstly and then evaluate the inference results on 5 different instruction test-sets. 

## Setup
- Set up the environment of [Alpaca](https://github.com/tatsu-lab/stanford_alpaca).
- pip install -e .  (Setup the truthfulqa environment)


## TruthfulQA test
We use standard TruthfulQA dataset to evaluate the performance of different models.
- test.sh: test the model's performance on TruthfulQA benchmark (Test the Hallucination)
- model names should use 'alpaca' as prefix
```
sh test_truthfulqa.sh
```


## Other tests
We use ChatGPT as the grader to evaluate the model's output.
```
export OPENAI_API_KEY
cd eval/
sh run_eval.sh
```


## References
- [Truthfulqa](https://github.com/sylinrl/TruthfulQA)
- [WizardLM](https://github.com/nlpxucan/WizardLM)
- [Koala](https://github.com/young-geng/EasyLM/tree/main)
- [Vicuna](https://vicuna.lmsys.org/)
- [GPT-4-Report](https://arxiv.org/pdf/2303.08774.pdf)