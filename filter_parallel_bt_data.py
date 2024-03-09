import json
from collections import defaultdict
split = 'train'
with open('./data/parallel_bt_{}.json'.format(split), encoding="utf-8") as file_obj:
	dataset = json.load(file_obj)
filtered_dataset = []
map = defaultdict(int)
for i in range(15):
	with open(f'./data/parallel_bt_{split}_reward_{i}.json', encoding="utf-8") as file_obj:
		contents = json.load(file_obj)
	for j in range(len(contents['reward12'])):
		# map[str(contents['reward12'])+' '+str(contents['reward3'])] += 1
		if contents['reward12'][j] >= 3 and contents['reward3'][j] >= 3:
			filtered_dataset.append(dataset[contents['idx'][j]])
print(len(filtered_dataset))
for k, v in map.items():
	print(k, v)
# with open(f'./data/parallel_bt_{split}_accurate.json', 'w', encoding="utf-8") as file_obj:
# 	json.dump(filtered_dataset, file_obj, ensure_ascii=False)