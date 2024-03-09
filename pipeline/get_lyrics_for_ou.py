# from gold_reference import Man, LetItGo, ForTheFirstTimeInForever, YouAreNotAlone, SixteenGoingOn17, TheImpossibleDream, MeAndTheSky, ExWives, Phantom, Matilda
# from gold_reference import AvenueQ, YouAreNotAlone, OnlyUs, IMissTheMountains, YouDontKnow, Popular, RevoltingChildren, Six, WavingThroughAWindow, Hurricane, WaitForIt	
from lyrics_info import Man, LetItGo, ForTheFirstTimeInForever, YouAreNotAlone, SixteenGoingOn17, TheImpossibleDream, MeAndTheSky, ExWives, Phantom, Matilda
from lyrics_info import AvenueQ, YouAreNotAlone, OnlyUs, IMissTheMountains, YouDontKnow, Popular, RevoltingChildren, Six, WavingThroughAWindow, Hurricane, WaitForIt	
from sacrebleu import corpus_bleu
from utils_pipe import extract_chinese, get_rhyme_id, get_rhyme_rule, get_syllable, count_length, count_syllables, cal_par_length
import re, json
def is_english(text):
	non_english = 0
	for char in text:
		if ord(char) > 127:
			non_english += 1
	if non_english*2.5 >= len(text):
		return False
	return True
songs = [AvenueQ, YouAreNotAlone, OnlyUs, IMissTheMountains, YouDontKnow, Popular, RevoltingChildren, Six, WavingThroughAWindow, Man, LetItGo, ForTheFirstTimeInForever, SixteenGoingOn17, TheImpossibleDream, MeAndTheSky, ExWives, Phantom, Matilda, Hurricane, WaitForIt]

dataset = []
dataset2 = ''

def add_item(en, zh, len_const):
	en = en.strip(',，。').replace('，',',').replace('’','\'')
	zh = zh.strip('，。')
	item = {'en': en, 'zh': zh, 'len': len_const}
	trans = zh
	for ch in ',./，。、！!?？：:':
		trans = trans.replace(ch, '')
	dataset.append(item)
	global dataset2
	dataset2 += en+'\n'

for songname in songs:
	# print(songs[idx].__name__)
	# if songs[idx].__name__ not in data:
	#	 print(songs[idx].__name__, "not in data", data.keys())
	gold_reference, syllables = songname()
	syllables = [[sum(x)] for x in syllables]
	paragraphs = re.split(r'\n\s*\n', gold_reference)
	print(songname.__name__, len(paragraphs))
	line_idx = 0
	for par in paragraphs:
		lines = [line for line in par.split('\n') if line.strip()!='']
		if len(lines)==0: continue
		print("len of lines = ", len(lines))
		for line in lines:
			dataset2 += line + '\n'
		# assert(len(lines) % 2==0)
		# bools = [is_english(line) for line in lines]
		# flag1, flag2 = [1]*(len(lines)//2)+[0]*(len(lines)//2)==bools, [1, 0]*(len(lines)//2)==bools
		# zh_lines = []
		# if flag1:
		# 	for i in range(len(lines)//2):
		# 		len_const = sum(count_syllables(lines[i])[0]) if line_idx >= len(syllables) else sum(syllables[line_idx])
		# 		add_item(lines[i], lines[i+len(lines)//2], len_const)
		# 		line_idx += 1
		# elif flag2:
		# 	for i in range(0, len(lines), 2):
		# 		len_const = sum(count_syllables(lines[i])[0]) if line_idx >= len(syllables) else sum(syllables[line_idx])
		# 		add_item(lines[i], lines[i+1], len_const)
		# 		line_idx += 1
		# else:
		# 	assert False
		dataset2 += '\n'

with open('test_set.txt', 'w') as f:
	f.write(dataset2)
# with open('test_set.json', 'w', encoding='utf-8') as file_obj:
	# json.dump(dataset, file_obj, indent=2, ensure_ascii=False)


# from pypinyin import lazy_pinyin
# import syllapy
# import re, os, sys, json
# import time
# from lyrics_info import Man, LetItGo, ForTheFirstTimeInForever, YouAreNotAlone, SixteenGoingOn17, TheImpossibleDream, MeAndTheSky, ExWives, Phantom, Matilda
# from lyrics_info import AvenueQ, YouAreNotAlone, OnlyUs, IMissTheMountains, YouDontKnow, Popular, RevoltingChildren, Six, WavingThroughAWindow, Hurricane, WaitForIt
# from utils_pipe import extract_chinese, get_rhyme_id, get_rhyme_rule, get_syllable, count_length, count_syllables, cal_par_length
# from openai import OpenAI
# import torch
# songs = [AvenueQ, YouAreNotAlone, OnlyUs, IMissTheMountains, YouDontKnow, Popular, RevoltingChildren, Six, WavingThroughAWindow, Man, LetItGo, ForTheFirstTimeInForever, SixteenGoingOn17, TheImpossibleDream, MeAndTheSky, ExWives, Phantom, Matilda, Hurricane, WaitForIt]

# for songname in songs:
# 	input_lyric, syllables = songname()
# 	pars = re.split(r'\n\s*\n', input_lyric)
# 	for par in pars:
# 		par_len = cal_par_length(par)
# 		idx = lst + par_len
# 		res, par_res = generate_par(par, syllables[lst: idx], use_kimi, use_reward, N, N2)
# 		result += res + '\n'
# 		gen_res.append(par_res)
# 		lst = idx
# 		print("par results is", res)
# 		#break