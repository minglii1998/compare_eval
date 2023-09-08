import json
import numpy as np
import os

review_dir = 'logs/xxx1-VS-xxx2'
review_file = os.path.join(review_dir,'vicuna_wrap_reviews.json')
raw_ins_file = 'test_data/vicuna_test_set.jsonl'

raw_ins = []
with open(raw_ins_file, 'r') as file:
    for line in file:
        raw_ins.append(json.loads(line))

with open(review_file, "r") as f:
    review_data = json.load(f)['data']

def get_scores_all(pure_data):
    score1, score2, score3 = 0, 0, 0

    k1_score = eval(pure_data['scores'])[0]
    k2_score = eval(pure_data['scores'])[1]
    k1_score_reverse = eval(pure_data['scores_reverse'])[1]
    k2_score_reverse = eval(pure_data['scores_reverse'])[0]

    if k1_score > k2_score and k1_score_reverse > k2_score_reverse:
        score1 += 1
    elif k1_score < k2_score and k1_score_reverse > k2_score_reverse:
        score2 += 1
    elif k1_score > k2_score and k1_score_reverse < k2_score_reverse:
        score2 += 1
    elif k1_score == k2_score and k1_score_reverse > k2_score_reverse:
        score1 += 1
    elif k1_score > k2_score and k1_score_reverse == k2_score_reverse:
        score1 += 1
    elif k1_score == k2_score and k1_score_reverse < k2_score_reverse:
        score3 += 1
    elif k1_score < k2_score and k1_score_reverse == k2_score_reverse:
        score3 += 1
    elif k1_score == k2_score and k1_score_reverse == k2_score_reverse:
        score2 += 1
    elif k1_score < k2_score and k1_score_reverse < k2_score_reverse:
        score3 += 1
        
    return [score1, score2, score3]

typr_dict = {}
for i in range(len(review_data)):
    type = raw_ins[i]['category']
    if type not in typr_dict.keys():
        typr_dict[type] = []
    scores = review_data[i]['scores']
    scores_reverse = review_data[i]['scores_reverse']
    score_list = get_scores_all(review_data[i])
    typr_dict[type].append(score_list)
    pass

for k in typr_dict.keys():
    lists = typr_dict[k]
    lists = np.array(lists)
    sum_score = lists.sum(0)
    avg_score = sum_score/sum_score.sum()
    print(k)
    print(sum_score)
    print(avg_score)

pass