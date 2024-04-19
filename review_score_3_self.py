import os
import json
import numpy as np
import matplotlib.pyplot as plt

scores_lists = [
    [90,67,61],
    [86,68,64],
    [96,68,54],
]
save_name = 'try1_'
key1 = 'Superfiltering model'
key2 = 'Full data model'
title_ = 'Model: LLaMA2-7B, Dataset: Alpaca' 

# scores_lists = [
#     [89,78,51],
#     [75,83,60],
#     [84,81,53],
# ]
# save_name = 'try2_'
# key1 = 'Superfiltering model'
# key2 = 'Full data model'
# title_ = 'Model: LLaMA2-13B, Dataset: Alpaca' 

# scores_lists = [
#     [69,83,66],
#     [71,90,57],
#     [78,79,61],
# ]
# save_name = 'try3_'
# key1 = 'Superfiltering model'
# key2 = 'Full data model'
# title_ = 'Model: LLaMA2-7B, Dataset: Alpaca-GPT4' 

# scores_lists = [
#     [70,87,61],
#     [67,94,57],
#     [81,73,64],
# ]
# save_name = 'try4_'
# key1 = 'Superfiltering model'
# key2 = 'Full data model'
# title_ = 'Model: LLaMA2-13B, Dataset: Alpaca-GPT4' 

model_names = [
    '5%',
    '10%',
    '15%',
]
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

    fig, ax = plt.subplots(figsize=(12, 3),dpi=300)
    plt.subplots_adjust()
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.8,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'black'
        ax.bar_label(rects, label_type='center', color=text_color, fontsize=12, fontweight='bold')

    for label in ax.get_yticklabels():
        label.set_size(12)
        label.set_weight('bold')  # Adjusting the y-axis label properties

    # ax.legend(ncols=3, loc='upper center', fontsize=10)

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

for i,score_i in enumerate(scores_lists):
    scores_all = score_i
    category_names = [f"{key1} wins", "Tie", f"{key2} wins"]
    results[model_names[i]] = scores_all

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
cropped_img = crop_edges(img_path,250,0,250,0)
cropped_img.save(img_path)
pass

