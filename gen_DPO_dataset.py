import json
import random
with open('llm_scores.json', encoding="utf-8") as file_obj:
    res_tot = json.load(file_obj)
with open('data_for_llm_scoring.json', encoding="utf-8") as file_obj:
    lyrics = json.load(file_obj)
# with open('translation_raw_data_for_DPO.json', encoding="utf-8") as file_obj:
#     translation = json.load(file_obj)
# with open('negative_data_for_DPO.json', encoding="utf-8") as file_obj:
#     negative = json.load(file_obj)

from llm_score import EvalReward
model = EvalReward()
num_en = 0
num_pairs = 0
data = []
for idx, row in enumerate(res_tot):
    en = row['en']
    pairs = []
    
    zh = []
    scores = []
    reward_scores = []
    for trans in lyrics[idx]['llama']:
        if trans in row['zh']:
            zh.append(trans)
            scores.append(row['scores'][row['zh'].index(trans)])
            reward_scores.append([model.eval_reward12('', en, trans)]) # , model.eval_reward3('', en, trans)
    num = len(zh)
    
    if num == 0:
        continue
    
    # big_thres = 0.9
    # small_thres = 0.5
    for i in range(num):
        for j in range(num):
            if reward_scores[i][0] > reward_scores[j][0]:
                pairs.append((i, j))
            # d1 = scores[i][0] - scores[j][0]
            # d2 = scores[i][1] - scores[j][1]
            # if d1 > small_thres and d2 > small_thres:
            #     pairs.append((i, j))
            # elif d1 < -small_thres and d2 < -small_thres:
            #     pairs.append((j, i))
            # elif d1 > 0 and d2 > big_thres:
            #     pairs.append((i, j))
            # elif d1 < 0 and d2 < -big_thres:
            #     pairs.append((j, i))
            # elif d1 > big_thres and d2 > 0:
            #     pairs.append((i, j))
            # elif d1 < -big_thres and d2 < 0:
            #     pairs.append((j, i))
    print(num, reward_scores, pairs)
    
    data_point = {}
    data_point['en'] = en
    data_point['zh'] = []
    for zh in row['zh']:
        tar = zh
        for ch in ',./;|`~!#*()&-_，。、？?":；：、-……——！~·':
            tar = tar.replace(ch, '')
        data_point['zh'].append(tar)
    data_point['pairs'] = pairs   
    
    # if len(negative[idx]['desired_gen']) > 0:
    #     for j in range(len(negative[idx]['desired_gen'])):
    #         if abs(len(negative[idx]['desired_gen'][j]) - len(data_point['zh'][0])) > 5:
    #             data_point['zh'].append(negative[idx]['desired_gen'][j])
    #             for i in range(int(num)): 
    #                 data_point['pairs'].append((i, len(data_point['zh']) - 1))
    
    # if len(translation[idx]['desired_gen']) > 0:
    #     for j in range(len(translation[idx]['desired_gen'])):
    #         if abs(len(translation[idx]['desired_gen'][j]) - len(translation[idx]['kimi_gen'][0])) > 5:
    #             data_point['zh'].append(translation[idx]['desired_gen'][j])
    #             for i in range(num): 
    #                 data_point['pairs'].append((i, len(data_point['zh']) - 1))
    if len(data_point['pairs']) > 0:
        num_en += 1
        data.append(data_point) 
        num_pairs += len(data_point['pairs'])
        
        # check
        for pair in data_point['pairs']:
            if pair[1] >= len(data_point['zh']):
                print("errrrrrrrrrrrrrrrrrrrrrrrrrrrrrr!")
                print(len(data_point['zh']), pair, data_point)
    if idx % 40 == 0:
        print(idx, len(res_tot))
        # break

print(num_pairs, num_en)

num_train = int(num_en * 0.9)
random.shuffle(data)
data_train = data[:num_train]
data_test = data[num_train:]

print("num_train = ", num_train)
num_train_pairs = 0
for data in data_train:
    num_train_pairs += len(data['pairs'])
print("num_train_pairs = ", num_train_pairs)


with open('llamaquality_train.json', 'w', encoding='utf-8') as file_obj:
    json.dump(data_train, file_obj, ensure_ascii=False)
with open('llamaquality_test.json', 'w', encoding='utf-8') as file_obj:
    json.dump(data_test, file_obj, ensure_ascii=False)