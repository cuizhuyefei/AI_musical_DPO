import os
import numpy as np
import json
import re

# read in ou_par_res.txt
ou_par_res = []
with open('ou_par_full_output.txt', 'r') as f:
    # split into paragraphs using '\n\n'
    paragraphs = f.read().split('\n\n')
    ou_par_res = paragraphs
    
# read in song_lyrics.json
song_lyrics = []
with open('final_experiments/song_lyrics.json', 'r') as f:
    song_lyrics = json.load(f)
    
par_idx = 0
res = {}

import sys
# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录
parent_dir = os.path.dirname(current_dir)
# 将上一级目录添加到 sys.path
sys.path.append(parent_dir)

from llm_score import EvalReward
reward = EvalReward()

for song in song_lyrics:
    song_name = song['name']
    
    paragraphs_zh_ref = re.split(r'\n\s*\n', song['gold_reference']) # split paragraphs using '\n\n'
    paragraphs_en = re.split(r'\n\s*\n', song['lyrics']) # split paragraphs using '\n\n'
    num_par = len(paragraphs_zh_ref) # get the number of paragraphs in the song
    assert num_par == len(paragraphs_en), f"Number of paragraphs in {song_name} is not the same in zh and en"
    
    res[song_name] = []
    for i in range(num_par):
        
        par_zh_trans = ou_par_res[par_idx]
        par_idx += 1
        
        # get ou's length constraints for this paragraph
        ou_length_constraint = []
        file_name = f'/usr0/home/zhuoruiy/Ou_backup/Dataset/datasets/real/full_testset/par_{par_idx}/constraints/source/test.target'
        with open(file_name, 'r') as f:
            # read each line
            lines = f.readlines()
            for line in lines:
                # extract the numbers from the line
                num1 = re.search(r'\d+', line).group()
                # print("num1", num1)
                ou_length_constraint.append(int(num1))
        
        par_res = []
        
        # for each line
        en_lines = [line for line in paragraphs_en[i].split('\n') if line.strip()!='']
        zh_trans_lines = [line for line in par_zh_trans.split('\n') if line.strip()!='']
        
        assert len(en_lines) == len(zh_trans_lines), f"Number of lines in paragraph {i+1} of {song_name} is not the same in zh and en"
        assert len(en_lines) == len(ou_length_constraint), f"Number of lines in paragraph {i+1} of {song_name} is not the same as length constraint"
        
        for j in range(len(en_lines)):
            en_line = en_lines[j]
            zh_trans_line = zh_trans_lines[j]
            en_par = paragraphs_en[i]
            reward12 = reward.eval_reward12(en_par, en_line, zh_trans_line)
            reward3 = reward.eval_reward3(en_par, en_line, zh_trans_line)
            
            par_res.append({
                "en_line": en_line,
                "zh_trans": [zh_trans_line],
                "all_zh_trans": [zh_trans_line],
                "par": en_par,
                "scores_of_all_trans": [],
                "len_constraint": ou_length_constraint[j], # TODO: replace with len_constraint
                "best_rhyme": None,
                "reward12": [reward12],
                "reward3": [reward3],
            })
        res[song_name].append(par_res)
    
# save res to ou_par_res.json, with chinese encoding
print('saving to final_experiments/ou_par_allres.json')
with open('final_experiments/ou_par_allres.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=4, ensure_ascii=False)

# # label song in ou_res in not generated
# label_file = 'ou_res_label.txt'
# if os.path.exists(label_file):
#     with open(label_file, 'r') as f:
#         label = f.read().split('\n')
#     label = [int(l) for l in label]
# else:
#     labels = []
#     idx = 0
#     while idx < len(ou_par_res):
#         print(f"Song {idx+1}/{len(ou_par_res)}:")
#         print(ou_par_res[idx])
        
#         print("Choose from the following songs:")
#         remaining_songs = []
#         for i in range(len(song_length)):
#             if i+1 in labels:
#                 continue
#             remaining_songs.append(f"{i+1}. {label_to_song[i+1]}")
#             # print(f"{i+1}. {label_to_song[i+1]}")
        
#         # print remaining songs in three column
#         num_songs = len(remaining_songs)
#         num_col = 3
#         num_row = int(np.ceil(num_songs/num_col))
#         for i in range(num_row):
#             for j in range(num_col):
#                 output_idx = i*num_col + j
#                 if output_idx < num_songs:
#                     print(remaining_songs[output_idx], end='\t')
#             print()
        
#         label = input("Which song is this?: ")
#         labels.append(int(label))
#         song = label_to_song[int(label)]
#         idx += song_length[song]
#         print("you have chosen song", song, "length of song is", song_length[song], "paragraphs")
#     with open(label_file, 'w') as f:
#         f.write('\n'.join(labels))