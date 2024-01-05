import os
import json
import numpy as np
import matplotlib.pyplot as plt

# review_home_path_list = [
#     'logs/wiz70_selection_pro_v4_ppl_sharegpt-VS-gpt4',
#     'logs/wiz70_selection_pro_v4_ppl_sharegpt-VS-Reflection_Alp_QA',
#     'logs/wiz70_selection_pro_v4_ppl_sharegpt-VS-selfee_7b',
#     'logs/wiz70_selection_pro_v4_ppl_sharegpt-VS-Llama_2_7b_chat',
#     'logs/wiz70_selection_pro_v4_ppl_sharegpt-VS-vicuna_7b_v1_5',
#     'logs/wiz70_selection_pro_v4_ppl_sharegpt-VS-chatgpt',
#     'logs/wiz70_selection_pro_v4_ppl_sharegpt-VS-bard',
#     'logs/wiz70_selection_pro_v4_ppl_sharegpt-VS-WizardLM_official_7b',
#     'logs/wiz70_selection_pro_v4_ppl_sharegpt-VS-alpaca_official_7b',
# ]
# model_names = [
#     'GPT4',
#     'Recycled Alpaca 7B',
#     'SelFee 7B',
#     'LLaMa3 Chat 7B',
#     'Vicuna 7B v1.5',
#     'WizardLM 7B',
#     'BARD',
#     'WizardLM 7B',
#     'Alpaca 7B',
# ]
# save_name = 'other_results/sRecycled_Alapca_7b'
# key1 = 'sRecycled Alpaca 7B'
# key2 = 'Other Models'
# title_ = key1 + ' vs. ' + key2 

review_home_path_list = [
    'logs/wiz70_selection_pro_v4_ppl_sharegpt-VS-gpt4',
    'logs/wiz70_selection_pro_v4_ppl_sharegpt-VS-Reflection_Wiz70_QA',
    'logs/wiz70_selection_pro_v4_ppl_sharegpt-VS-selfee_7b',
    'logs/wiz70_selection_pro_v4_ppl_sharegpt-VS-Llama_2_7b_chat',
    'logs/wiz70_selection_pro_v4_ppl_sharegpt-VS-vicuna_7b_v1_5',
    'logs/wiz70_selection_pro_v4_ppl_sharegpt-VS-chatgpt',
    'logs/wiz70_selection_pro_v4_ppl_sharegpt-VS-bard',
    'logs/wiz70_selection_pro_v4_ppl_sharegpt-VS-WizardLM_official_7b',
    'logs/wiz70_selection_pro_v4_ppl_sharegpt-VS-alpaca_official_7b',
]
model_names = [
    'GPT4',
    'Recycled WizardLM 7B',
    'SelFee 7B',
    'LLaMa3 Chat 7B',
    'Vicuna 7B v1.5',
    'WizardLM 7B',
    'BARD',
    'WizardLM 7B',
    'Alpaca 7B',
]
save_name = 'other_results/sRecycled_WizardLM_7b'
key1 = 'sRecycled WizardLM 7B'
key2 = 'Other Models'
title_ = key1 + ' vs. ' + key2 

# review_home_path_list = [
#     'logs/Reflection_Wiz70_QA-VS-gpt4',
#     'logs/Reflection_Wiz70_QA-VS-selfee_7b',
#     'logs/Reflection_Wiz70_QA-VS-Llama_2_7b_chat',
#     'logs/Reflection_Wiz70_QA-VS-vicuna_7b_v1_5',
#     'logs/Reflection_Wiz70_QA-VS-chatgpt',
#     'logs/Reflection_Wiz70_QA-VS-bard',
#     'logs/Reflection_Wiz70_QA-VS-WizardLM_official_7b',
#     'logs/Reflection_Wiz70_QA-VS-alpaca_official_7b',
# ]
# model_names = [
#     'GPT4',
#     'SelFee 7B',
#     'LLaMA2 Chat 7B',
#     'Vicuna 7B',
#     'ChatGPT',
#     'BARD',
#     'WizardLM 7B',
#     'Alpaca 7B',
# ]
# save_name = 'other_results/other_results_wiz'
# key1 = 'Recycled WizardLM 7B'
# key2 = 'Other Models'
# title_ = key1 + ' vs. ' + key2 


datasets = ['Vicuna']
def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlGn'](
        np.linspace(0.2, 0.8, data.shape[1]))

    fig, ax = plt.subplots(figsize=(14, 10),dpi=300)
    plt.subplots_adjust(left=0.4)
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.6,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'black'
        ax.bar_label(rects, label_type='center', color=text_color, fontsize=12, fontweight='bold')

    for label in ax.get_yticklabels():
        label.set_size(12)
        label.set_weight('bold')  # Adjusting the y-axis label properties

    ax.legend(ncols=1, loc='upper right', fontsize=12)

    return fig, ax


results = {}
def get_scores_all(pure_data):
    score1, score2, score3 = 0, 0, 0
    l = len(pure_data)
    for i in range(l):
        k1_score = eval(pure_data[i]['scores'])[0]
        k2_score = eval(pure_data[i]['scores'])[1]
        k1_score_reverse = eval(pure_data[i]['scores_reverse'])[1]
        k2_score_reverse = eval(pure_data[i]['scores_reverse'])[0]

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

for i,review_home_path in enumerate(review_home_path_list):
    scores_all = [0,0,0]
    for dataset in datasets:
        review_path = ''
        for root, ds, fs in os.walk(review_home_path):
                for f in fs:
                    # if 'reviews' in f and f.endswith('.json') and dataset.lower() in f:
                    #     review_path = os.path.join(root, f)
                    if 'reviews_gpt4' in f and f.endswith('.json') and dataset.lower() in f:
                        review_path = os.path.join(root, f)
        with open(review_path, "r") as f:
            review_data = json.load(f)
        pure_data = review_data['data']

        scores = get_scores_all(pure_data)
        for jj in range(len(scores)):
            scores_all[jj] += scores[jj]
        pass
    category_names = [f"{key1} wins", "Tie", f"{key2} wins"]
    results[model_names[i]] = scores_all

def cal_rate(results):
    win = 0
    tie = 0
    loss = 0
    for k in results.keys():
        win += results[k][0]
        tie += results[k][1]
        loss += results[k][2]
    print((win-loss)/(win+loss+tie)+1)

cal_rate(results)
survey(results, category_names)
img_path = os.path.join(save_name+'.jpg')
ax = plt.gca()
ax.set_title(title_, fontsize=20)
plt.savefig(img_path)
pass

from PIL import Image
def crop_edges(image_path, left, upper, right, lower):
    with Image.open(image_path) as img:
        width, height = img.size
        cropped = img.crop((left, upper, width - right, height - lower))
        return cropped
cropped_img = crop_edges(img_path,50,150,150,150)
cropped_img.save(img_path)
pass

