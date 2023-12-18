import matplotlib
matplotlib.use('Agg')  # 使用Agg后端
import numpy as np
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import json

def cal(list1, list2, title='Scatter Plot of Two Lists', xlabel='X-axis', ylabel='Y-axis', figname='scatter_plot.png'):
    n = len(list1)
    for i in range(n):
        print(list1[i], list2[i])
    
    # 皮尔逊相关系数
    pearson_corr = np.corrcoef(list1, list2)[0, 1]
    print(f"Pearson Correlation: {pearson_corr}")

    # 斯皮尔曼秩相关系数
    spearman_corr, _ = spearmanr(list1, list2)
    print(f"Spearman Rank Correlation: {spearman_corr}")

    # 肯德尔秩相关系数
    kendall_corr, _ = kendalltau(list1, list2)
    print(f"Kendall Rank Correlation: {kendall_corr}")

    data = np.zeros((4, 4))
    for i in range(n):
        data[3 - int(list1[i] - 1)][int(list2[i] - 1)] += 1
    # 创建一个颜色映射，数值越大颜色越浓
    colors = plt.cm.Purples(np.sqrt(data) / np.sqrt(np.max(data)) * 0.55 + 0.45)
    # 绘制图形
    fig, ax = plt.subplots()
    ax.imshow(colors, cmap='Purples')

    # 在每个格子中添加数值标签
    for i in range(4):
        for j in range(4):
            ax.text(j, i, data[i, j], ha='center', va='center', color='w')
    
    for i in range(4):
        ax.text(i, 3.65, str(i + 1), ha='center', va='center', color='k')
    ax.text(1.5, 3.8, xlabel, ha='center', va='center', color='k')
    for i in range(4):
        ax.text(-0.65, i, str(4 - i), ha='center', va='center', color='k', rotation=90)
    ax.text(-0.8, 1.5, ylabel, ha='center', va='center', color='k', rotation=90)
    
    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(figname)


if __name__ == '__main__':
    # list1 = [1, 2, 3, 4, 5]
    # list2 = [5, 4, 3, 2, 1]
    
    with open('human_scores_kimi_data.json', encoding="utf-8") as file_obj: # human_scores_kimi_data
        data = json.load(file_obj)
    print("len of human data", len(data))
    list0 = []
    for data_point in data:
        # print(len(data_point['scores']))
        for scores in data_point['human_score']:
            list0.append(scores[0])
    
    with open('gpt4_scores_kimi_data.json', encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    print("len of gpt4 data", len(data))
    list1 = []
    for data_point in data:
        # print(len(data_point['scores']))
        for scores in data_point['scores']:
            list1.append(scores[0])
    
    with open('llama_scores_kimi_data.json', encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    print("len of llama data", len(data))
    list2 = []
    list3 = []
    for data_point in data:
        if len(data_point['llama_1']) > 3:
            print(data_point)
        # print(len(data_point['llama_1']))
        for scores in data_point['llama_1']:
            list2.append(scores[0])
            list3.append(scores[1])
    
    print(len(list0), len(list1), len(list3))
    assert(len(list1) == len(list3))
    # cal(list1, list2, 'gpt4 vs llama scores on aspect1', 'gpt4 scores', 'llama scores', 'gpt4_llama_scores_aspect1.png')
    # cal(list0, list1, 'human vs gpt4 scores on aspect1', 'human scores', 'gpt4 scores', 'human_gpt4_scores_aspect1.png')
    cal(list0, list2, 'human vs llama scores on aspect1', 'human scores', 'llama scores', 'human_llama_1_aspect1.png')
    