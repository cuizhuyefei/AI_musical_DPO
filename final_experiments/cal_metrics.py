import sys 
sys.path.append(".")
from llm_score import llamaGenAPI
from llm_score import EvalReward
# from pipeline.utils_pipe import get_rhyme_id
import syllapy
import json, re
from collections import Counter

# from pipeline.gold_reference import Man, LetItGo, ForTheFirstTimeInForever, YouAreNotAlone, SixteenGoingOn17, TheImpossibleDream, MeAndTheSky, ExWives, Phantom, Matilda
# from pipeline.gold_reference import AvenueQ, YouAreNotAlone, OnlyUs, IMissTheMountains, YouDontKnow, Popular, RevoltingChildren, Six, WavingThroughAWindow, Hurricane, WaitForIt

# with open('llama_songs_allres_N20.json', encoding="utf-8") as file_obj:
#     ou_ref = json.load(file_obj)
# with open('ou_smalltest_par_res.txt', 'r', encoding='utf8') as f:
#     zh_lines = f.readlines()
# with open('pipeline/test_set3.txt', 'r', encoding='utf8') as f:
#     en_lines = f.readlines()
# assert(len(zh_lines) == len(en_lines))
# zh_lines = [line.strip() for line in zh_lines if len(line.strip()) > 0]
# print("deal with ou's data len =", len(zh_lines))
# ou = {}
# idx = 0
# songs = ['AvenueQ', 'YouAreNotAlone', 'OnlyUs', 'IMissTheMountains', 'YouDontKnow', 'Popular', 'RevoltingChildren', 'Six', 'WavingThroughAWindow', 'Man', 'LetItGo', 'ForTheFirstTimeInForever', 'SixteenGoingOn17', 'TheImpossibleDream', 'MeAndTheSky', 'ExWives', 'Phantom', 'Matilda', 'Hurricane', 'WaitForIt']
# reward = EvalReward()
# for song in songs:
#     song_res = []
#     for par in ou_ref[song]:
#         par_res = []
#         for line in par:
#             en_line = line['en_line'].strip(',，。').replace('，',',').replace('’','\'')
#             en_line = en_line.strip()
#             zh_line = zh_lines[idx]
#             idx += 1
#             line_res = {
#                 'en_line': en_line,
#                 'zh_trans': [zh_line],
#                 'all_zh_trans': [zh_line],
#                 'par': line['par'],
#                 'len_constraint': line['len_constraint'],
#                 'best_rhyme': -1,
#                 'rhyme_type': [get_rhyme_id(zh_line)],
#                 'reward12': [reward.eval_reward12(line['par'], en_line, zh_line)],
#                 'reward3': [reward.eval_reward3(line['par'], en_line, zh_line)]
#             }
#             par_res.append(line_res)
#         song_res.append(par_res)
#     ou[song] = song_res
# with open('ou_smalltest_par_res.json', 'w', encoding='utf-8') as file_obj:
#     json.dump(ou, file_obj, ensure_ascii=False)
# exit(0)

use_reward = False

def choose_best(line_res, is_end):
    def align_score(gt, y):
        return gt-y if gt>y else 2*(y-gt)
    def count_length(str):
        # str = str.strip()
        for ch in " ，。！；？,.!;?()-：:”“…——、":    
            str = str.replace(ch, '')
        return len(str)
    def word_boundary_score(str, trans):
        def penalty(x):
            return x if x>0 else -2*x
        def get_syllable(s): # wrong cases: debauched 2, bravely 2
            ret = [syllapy.count(ss) for ss in s.split(' ')]
            return sum(ret)
        # print("original str = ", str)
        for ch in "，。！；？,.!;?()-：:”“…——、":    
            str = str.replace(ch, ',')
            trans = trans.replace(ch, ' ')
        # print("after replace str=", str)
        str = str.split(',')
        trans = trans.split(' ')
        ref = [get_syllable(ss) for ss in str if len(ss) > 0]
        # print("ref = ", ref, str)
        s = [len(ss) for ss in trans if len(ss) > 0]
        if len(ref) > len(s):
            return 100000
        f = [[100000 for i in range(len(s))] for j in range(len(ref))]
        for i in range(len(ref)):
            for j in range(len(s)):
                if i == 0:
                    f[i][j] = penalty(ref[i] - sum(s[:j+1]))
                elif j < i:
                    f[i][j] = 100000
                else:
                    f[i][j] = min(f[i-1][k] + penalty(ref[i] - sum(s[k+1:j+1])) for k in range(j))
        score = f[len(ref)-1][len(s)-1]
        return score
    if use_reward:
        scores = [3*align_score(line_res['len_constraint'], count_length(line_res['all_zh_trans'][j]))
            -(2 if line_res['rhyme_type'][j]==line_res['best_rhyme'] else 0)
            +(10**10 if is_end and line_res['rhyme_type'][j]!=line_res['best_rhyme'] else 0)
            -(1*line_res['reward12'][j])
            +(10**10 if line_res['reward12'][j]<=2 else 0)
            -(line_res['reward3'][j]) for j in range(len(line_res['all_zh_trans']))]
        # scores = [3*word_boundary_score(line_res['en_line'], line_res['all_zh_trans'][j])
        #     -(2 if line_res['rhyme_type'][j]==line_res['best_rhyme'] else 0)
        #     +(10**10 if is_end and line_res['rhyme_type'][j]!=line_res['best_rhyme'] else 0)
        #     -(1*line_res['reward12'][j])
        #     +(10**10 if line_res['reward12'][j]<=2 else 0)
        #     -(line_res['reward3'][j]) for j in range(len(line_res['all_zh_trans']))]
    else:
        scores = [3*align_score(line_res['len_constraint'], count_length(line_res['all_zh_trans'][j]))
            +(10**10 if is_end and line_res['rhyme_type'][j]!=line_res['best_rhyme'] else 0)
            -(2 if line_res['rhyme_type'][j]==line_res['best_rhyme'] else 0) for j in range(len(line_res['all_zh_trans']))]
    # if scores != line_res['scores_of_all_trans']:
    #     print(scores, line_res['scores_of_all_trans'])
    #     print([3*align_score(line_res['len_constraint'], count_length(line_res['all_zh_trans'][j])) for j in range(len(line_res['all_zh_trans']))])
    #     print([-(2 if line_res['rhyme_type'][j]==line_res['best_rhyme'] else 0) for j in range(len(line_res['all_zh_trans']))])
    #     print([-(line_res['reward12'][j])
    #     +(10 if line_res['reward12'][j]<=2 else 0)
    #     -(line_res['reward3'][j]) for j in range(len(line_res['all_zh_trans']))])
    # assert(scores == line_res['scores_of_all_trans'])
    return line_res['all_zh_trans'][scores.index(min(scores))]

# def get_line_syllables(line):
#     syllable = 0
#     parts = line.replace('\r', '').replace('|', ',').split(',')
#     for part in parts:
#         for s in part.strip('.!?').split(' '):
#             if len(s)==0: continue
#             syllable += get_syllable(s)
#     return syllable

def cal_length_acc(data):
    cor = 0
    err = 0
    num = 0
    for song in data:
        song_res = data[song]
        for par_res in song_res:
            for i, line_res in enumerate(par_res):
                line_res['zh_trans'][0] = choose_best(line_res, i == len(par_res) - 1) # 判定line_res是否is_end
                desired_length = int(line_res['len_constraint'])
                # desired_length = get_line_syllables(line_res['en_line'])
                trans = line_res['zh_trans'][0].strip()
                for ch in ',./，。、！!?？：:':
                    trans = trans.replace(ch, '')
                real_length = len(trans)
                cor += int(real_length == desired_length)
                err += abs(real_length - desired_length)
                # if real_length != desired_length:
                #     print(line_res['en_line'])
                num += 1
    return cor / num, err / num

def cal_reward(data):
    if not use_reward:
        return 0, 0
    sum12 = 0
    sum3 = 0
    num = 0
    for song in data:
        song_res = data[song]
        for par_res in song_res:
            for i, line_res in enumerate(par_res):
                line_res['zh_trans'][0] = choose_best(line_res, i == len(par_res) - 1) # 判定line_res是否is_end
                idx = line_res['all_zh_trans'].index(line_res['zh_trans'][0])
                sum12 += int(line_res['reward12'][idx])
                sum3 += int(line_res['reward3'][idx])
                num += 1
    return sum12 / num, sum3 / num

def cal_rhyme(data):
    score = 0
    num = 0
    for song in data:
        song_res = data[song]
        for par_res in song_res:
            rhymes = []
            for i, line_res in enumerate(par_res):
                line_res['zh_trans'][0] = choose_best(line_res, i == len(par_res) - 1) # 判定line_res是否is_end
                idx = line_res['all_zh_trans'].index(line_res['zh_trans'][0])
                rhymes.append(line_res['rhyme_type'][idx])
            counter = Counter(rhymes)
            most_common_number, count = counter.most_common(1)[0]
            score += count / len(rhymes)
            num += 1
            # score += count
            # num += len(rhymes)
    return score / num

def cal_best_data(data):
    cor = 0
    err = 0
    num = 0
    res = []
    for song in data:
        song_res = data[song]
        res_song = []
        for par_res in song_res:
            res_par = []
            for i, line_res in enumerate(par_res):
                res_line = {}
                line_res['zh_trans'][0] = choose_best(line_res, i == len(par_res) - 1) # 判定line_res是否is_end
                res_line['en_line'] = line_res['en_line']
                res_line['zh_trans'] = line_res['zh_trans'][0]
                res_par.append(res_line)
                num += 1
            res_song.append(res_par)
        res.append(res_song)
    return res

# from bert_score import BERTScorer
# scorer = BERTScorer(lang="zh")
from comet import download_model, load_from_checkpoint
model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)
def cal_bleu(data):
    from sacrebleu import corpus_bleu
    def is_english(text):
        non_english = 0
        for char in text:
            if ord(char) > 127:
                non_english += 1
        if non_english*2.5 >= len(text):
            return False
        return True
    with open('final_experiments/song_lyrics.json', encoding="utf-8") as file_obj:
        songs = json.load(file_obj)
    ens = []
    preds = []
    refs = []
    for idx, song in enumerate(data):
        # print(songs[idx].__name__)
        # if songs[idx].__name__ not in data:
        #     print(songs[idx].__name__, "not in data", data.keys())
        song_res = data[songs[idx]["name"]]
        gold_reference = songs[idx]["gold_reference"]
        paragraphs = re.split(r'\n\s*\n', gold_reference)
        # print(song_res)
        # print(paragraphs)
        # print(songs[idx].__name__, len(paragraphs), len(song_res))
        for jdx, par_res in enumerate(song_res):
            zh_lines = [line for line in paragraphs[jdx].split('\n') if line.strip()!='']
            assert(len(par_res) == len(zh_lines))
            for i, line_res in enumerate(par_res):
                line_res['zh_trans'][0] = choose_best(line_res, i == len(par_res) - 1) # 判定line_res是否is_end
                ens.append(line_res['en_line'][0])
                preds.append(line_res['zh_trans'][0])
                refs.append(zh_lines[i])
    bleu = round(corpus_bleu(preds, [refs], tokenize='zh').score, 2)
    data_for_comet = [{"src": ens[i], "mt": preds[i], "ref": refs[i]} for i in range(len(ens))]
    comet = model.predict(data_for_comet, batch_size=32, gpus=1)
    return bleu, round(comet.system_score, 4)

def cal_all_metrics(data):
    bleu, comet = cal_bleu(data)
    len_acc, len_err = cal_length_acc(data)
    reward12, reward3 = cal_reward(data)
    rhyme = cal_rhyme(data)
    if len(data) < 56: print(f'[only {len(data)} songs]', end='')
    print("len_acc, reward12, reward3, rhyme, bleu, comet =", round(len_acc,3), round(reward12,3), round(reward3,3), round(rhyme,3), bleu, comet)
    # return len_acc, len_err, reward12, reward3, rhyme
    new_best_data = cal_best_data(data)
    return new_best_data

if __name__ == '__main__':
    with open('final_experiments/llama_songs_allres_N40+40_ckpt3.json', encoding="utf-8") as file_obj:
        llama = json.load(file_obj)
    print("for N40+40_ckpt3:")
    cal_all_metrics(llama)
    with open('final_experiments/llama_songs_allres_N40+40_ckpt2.json', encoding="utf-8") as file_obj:
        llama = json.load(file_obj)
    print("for N40+40_ckpt2:")
    cal_all_metrics(llama)
    with open('final_experiments/llama_songs_allres_N40+40_ckpt1.json', encoding="utf-8") as file_obj:
        llama = json.load(file_obj)
    print("for N40+40_ckpt1:")
    cal_all_metrics(llama)
    
    with open('final_experiments/llama_songs_allres_N80+0_ckpt3.json', encoding="utf-8") as file_obj:
        llama = json.load(file_obj)
    print("for N80+0_ckpt3:")
    cal_all_metrics(llama)
    with open('final_experiments/llama_songs_allres_N80+80_ckpt3.json', encoding="utf-8") as file_obj:
        llama = json.load(file_obj)
    print("for N80+80_ckpt3:")
    cal_all_metrics(llama)
    with open('final_experiments/llama_songs_allres_N40+40_ckpt0.json', encoding="utf-8") as file_obj:
        llama = json.load(file_obj)
    print("for N40+40_ckpt0:")
    cal_all_metrics(llama)
    