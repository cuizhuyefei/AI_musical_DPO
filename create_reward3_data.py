import json
import random, os
# with open('llm_scores.json', encoding="utf-8") as file_obj:
#     res_tot = json.load(file_obj)
# with open('data_for_llm_scoring.json', encoding="utf-8") as file_obj:
#     lyrics = json.load(file_obj)
# with open('translation_raw_data_for_DPO.json', encoding="utf-8") as file_obj:
#     translation = json.load(file_obj)
# with open('negative_data_for_DPO.json', encoding="utf-8") as file_obj:
#     negative = json.load(file_obj)

num_en = 0
num_pairs = 0
tot = 0
data = []
for i in range(5):
	if not os.path.exists(f'data/data_scores_{i}.json'):
		continue
	with open(f'data/data_scores_{i}.json', encoding="utf-8") as file_obj: #read
		score = json.load(file_obj)
	with open(f'data/scoring_data_{i}.json', encoding="utf-8") as file_obj: #read
		dataset = json.load(file_obj)
	
	idx = 0
	while idx < len(dataset):
		# jdx = idx + len(dataset[idx]['all_zh_trans']) - 1
		jdx = idx
		while jdx+1 < len(dataset) and dataset[jdx+1]['en_line'] == dataset[idx]['en_line']:
			jdx += 1
		data_item = {"en": dataset[idx]['en_line'], "zh": [], "pairs": []}
		for k in range(idx, jdx+1):
			data_item["zh"].append(dataset[k]['zh_line'].replace('\n',''))
		for i in range(idx, jdx+1):
			for j in range(idx, jdx+1):
				if score[i][2] > score[j][2]:
					tot += 1
					data_item["pairs"].append((i-idx, j-idx))
		if len(data_item["pairs"])>0:
			data.append(data_item)
		idx = jdx + 1
	# for idx, row in enumerate(dataset):
	# 	en = row['en_line']
	# 	zh = row['zh_line']
	# 	pairs = []
		
	# 	zh = []
	# 	scores = []
	# 	for trans in lyrics[idx]['llama']:
	# 		if trans in row['zh']:
	# 			zh.append(trans)
	# 			scores.append(row['scores'][row['zh'].index(trans)])
	# 	num = len(zh)
		
	# 	if num == 0:
	# 		continue
		
	# 	big_thres = 0.9
	# 	small_thres = 0.5
	# 	for i in range(num):
	# 		for j in range(i):
	# 			d1 = scores[i][0] - scores[j][0]
	# 			d2 = scores[i][1] - scores[j][1]
	# 			if d1 > small_thres and d2 > small_thres:
	# 				pairs.append((i, j))
	# 			elif d1 < -small_thres and d2 < -small_thres:
	# 				pairs.append((j, i))
	# 			elif d1 > 0 and d2 > big_thres:
	# 				pairs.append((i, j))
	# 			elif d1 < 0 and d2 < -big_thres:
	# 				pairs.append((j, i))
	# 			elif d1 > big_thres and d2 > 0:
	# 				pairs.append((i, j))
	# 			elif d1 < -big_thres and d2 < 0:
	# 				pairs.append((j, i))
		
	# 	data_point = {}
	# 	data_point['en'] = en
	# 	data_point['zh'] = []
	# 	for zh in row['zh']:
	# 		tar = zh
	# 		for ch in ',./;|`~!#*()&-_，。、？?":；：、-……——！~·':
	# 			tar = tar.replace(ch, '')
	# 		data_point['zh'].append(tar)
	# 	data_point['pairs'] = pairs   
		
	# 	# if len(negative[idx]['desired_gen']) > 0:
	# 	#     for j in range(len(negative[idx]['desired_gen'])):
	# 	#         if abs(len(negative[idx]['desired_gen'][j]) - len(data_point['zh'][0])) > 5:
	# 	#             data_point['zh'].append(negative[idx]['desired_gen'][j])
	# 	#             for i in range(int(num)): 
	# 	#                 data_point['pairs'].append((i, len(data_point['zh']) - 1))
		
	# 	# if len(translation[idx]['desired_gen']) > 0:
	# 	#     for j in range(len(translation[idx]['desired_gen'])):
	# 	#         if abs(len(translation[idx]['desired_gen'][j]) - len(translation[idx]['kimi_gen'][0])) > 5:
	# 	#             data_point['zh'].append(translation[idx]['desired_gen'][j])
	# 	#             for i in range(num): 
	# 	#                 data_point['pairs'].append((i, len(data_point['zh']) - 1))
	# 	if len(data_point['pairs']) > 0:
	# 		num_en += 1
	# 		data.append(data_point) 
	# 		num_pairs += len(data_point['pairs'])
			
	# 		# check
	# 		for pair in data_point['pairs']:
	# 			if pair[1] >= len(data_point['zh']):
	# 				print("errrrrrrrrrrrrrrrrrrrrrrrrrrrrrr!")
	# 				print(len(data_point['zh']), pair, data_point)


with open('data/reward3_data.json', 'w', encoding='utf-8') as file_obj:
    json.dump(data, file_obj, ensure_ascii=False)
num_pairs = tot
num_en = len(data)
print(num_pairs, num_en)

num_train = int(num_en * 0.9)
random.shuffle(data)
data_train = data[:num_train]
data_test = data[num_train:]

# print("num_train = ", num_train)
# num_train_pairs = 0
# for data in data_train:
#     num_train_pairs += len(data['pairs'])
# print("num_train_pairs = ", num_train_pairs)


# with open('data/reward3_data_train.json', 'w', encoding='utf-8') as file_obj:
#     json.dump(data_train, file_obj, ensure_ascii=False)
# with open('data/reward3_data_test.json', 'w', encoding='utf-8') as file_obj:
#     json.dump(data_test, file_obj, ensure_ascii=False)

tot = 0
for x in data_train:
	tot += len(x['pairs'])
print(tot)

tot = 0
for x in data_test:
	tot += len(x['pairs'])
print(tot)

# 把所有pair打印到.txt文件里
with open('data/reward3_data_train.txt', 'w', encoding='utf-8') as file_obj:
    res = ''
    for data in data_train:
        for pair in data['pairs']:
            res += data['en'] + '->' + data['zh'][pair[0]] + '\t' + data['en'] + '->' + data['zh'][pair[1]] + '\n'
    file_obj.write(res)

with open('data/reward3_data_test.txt', 'w', encoding='utf-8') as file_obj:
    res = ''
    for data in data_test:
        for pair in data['pairs']:
            res += data['en'] + '->' + data['zh'][pair[0]] + '\t' + data['en'] + '->' + data['zh'][pair[1]] + '\n'
    file_obj.write(res)
