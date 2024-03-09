import json, random, os
files = [
	'ou_smalltest_par_res.json', # ou
	'llama_v18_songs_allres_N40+40.json', # llama_40+40 T=0.6 top_p=0.95 (175w C)
	'llama_v19_songs_allres_N40+40.json', # llama_40+40 T=0.6 top_p=0.95 (175w filtered C) (ongoing)
	'llama_v17_songs_allres_N40+40.json', # llama_40+40 T=0.6 top_p=0.95 (175w filtered C + 70w acc_finetune)
	'kimi_songs_allres_N40+40.json', # kimi
	'llama_v33_songs_allres_N40+40.json',
]
par_contents = []
line_contents = []

with open(files[0], encoding="utf-8") as file_obj:
	data = json.load(file_obj)
for song in data:
	song_res = data[song]
	res_song = []
	for i, par_res in enumerate(song_res):
		res_par = []
		par_contents.append((song, i))
		for j, line_res in enumerate(par_res):
			line_contents.append((song, i, j, line_res['par']))

print(len(par_contents), len(line_contents))
# exit(0)

random.seed(36)
random.shuffle(par_contents)
random.shuffle(line_contents)
par_contents = par_contents#[:20]
line_contents = line_contents#[:50]
# print(par_contents, '\n', line_contents)

datas = []
for file in files:
	with open(file, encoding="utf-8") as file_obj:
		datas.append(json.load(file_obj))

par_output = []
for par in par_contents:
	par_item = []
	for idx, data in enumerate(datas):
		en = []
		zh = []
		# if idx==1:
		# 	print(par, data[par[0]][par[1]])
		for jdx, line in enumerate(data[par[0]][par[1]]):
			# print(jdx, line)
			en.append(line['en_line'])
			zh.append(line['zh_trans'][0])
		par_item.append({'file': files[idx], 'en': en, 'zh': zh})
	par_output.append(par_item)

line_output = []
for line in line_contents:
	line_item = []
	for idx, data in enumerate(datas):
		en = data[line[0]][line[1]][line[2]]['en_line']
		zh = data[line[0]][line[1]][line[2]]['zh_trans'][0]
		line_item.append({'file': files[idx], 'en': en, 'zh': zh, 'par': line[3]})
	line_output.append(line_item)

total_output = (par_output, line_output)
with open('big_cherry_pick.json', 'w', encoding='utf-8') as file_obj:
	json.dump(total_output, file_obj, indent=2, ensure_ascii=False)
