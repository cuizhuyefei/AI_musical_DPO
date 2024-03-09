from llm_score import llamaGenAPI
from llm_score import EvalReward
from pipeline.utils_pipe import get_rhyme_id
import json
from collections import Counter

with open('kimi_songs_allres_N20.json', encoding="utf-8") as file_obj:
    kimi20 = json.load(file_obj)
with open('kimi_songs_allres_N40.json', encoding="utf-8") as file_obj:
    kimi40 = json.load(file_obj)
with open('llama_songs_allres_N20.json', encoding="utf-8") as file_obj:
    llama20 = json.load(file_obj)
with open('llama_songs_allres_N40.json', encoding="utf-8") as file_obj:
    llama40 = json.load(file_obj)
with open('llama_songs_allres_N100.json', encoding="utf-8") as file_obj:
    llama100 = json.load(file_obj)
with open('ou_songs_allres.json', encoding="utf-8") as file_obj:
    ou = json.load(file_obj)


def compare(data_list):
    num = len(data_list)
    res = []
    for song in data_list[0]:
        num_par = len(data_list[0][song])
        for par_idx in range(num_par):
            num_line = len(data_list[0][song][par_idx])
            print("en par = ", data_list[0][song][par_idx][0]['par'])
            en_par = data_list[0][song][par_idx][0]['par']
            zh_par = []
            for idx in range(num):
                print("idx = ", idx)
                par_trans = ''
                for line_idx in range(num_line):
                    print(data_list[idx][song][par_idx][line_idx]['zh_trans'][0])
                    par_trans += data_list[idx][song][par_idx][line_idx]['zh_trans'][0] + '\n'
                zh_par.append(par_trans)
            res.append({'en_par': en_par, 'zh_par': zh_par})
    return res

if __name__ == '__main__':
    res = compare([kimi20, kimi40, llama20, llama40, llama100, ou])
    with open('for_cherry_pick.json', 'w', encoding='utf-8') as file_obj:
        json.dump(res, file_obj, ensure_ascii=False)