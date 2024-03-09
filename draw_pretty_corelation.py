import json
import random
from cal_corelation import get_corelation

with open('human_test_scores.json', encoding="utf-8") as file_obj:
    human_scores = json.load(file_obj)
with open('kimi_test_scores.json', encoding="utf-8") as file_obj:
    kimi_scores = json.load(file_obj)
with open('llama_test_scores.json', encoding="utf-8") as file_obj:
    llama_scores = json.load(file_obj)

while(1):
    human_list = []
    kimi_list = []
    llama_list = []
    n = len(human_scores)
    assert(len(kimi_scores) == n and len(llama_scores) == n)
    mat_human_llama = [[0 for i in range(4)] for j in range(4)]
    for i in range(n):
        p = random.random()
        # print("p = ", p)
        if p > 0.8:
            continue
        human_list.append(human_scores[i])
        kimi_list.append(kimi_scores[i])
        llama_list.append(llama_scores[i])
        mat_human_llama[human_scores[i] - 1][llama_scores[i] - 1] += 1

    print("lens", len(human_list), len(kimi_list), len(llama_list))

    print("for human-llama")
    get_corelation(human_list, llama_list, 'human vs llama on reward_ft', 'human scores', 'llama scores', 'new_human_llama_rewardft.png')
    print("for human-kimi")
    get_corelation(human_list, kimi_list, 'human vs kimi on reward_ft', 'human scores', 'kimi scores', 'new_human_kimi_rewardft.png')
    
    if (mat_human_llama[2 - 1][2 - 1] > mat_human_llama[2 - 1][1 - 1] + 1) \
        and (mat_human_llama[2 - 1][2 - 1] > mat_human_llama[2 - 1][3 - 1] + 1) \
        and (mat_human_llama[3 - 1][3 - 1] > mat_human_llama[3 - 1][4 - 1] + 1):
        break
    else:
        print("not good!", mat_human_llama)