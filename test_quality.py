import re, os, json
import time
from openai import OpenAI
from llm_score import EvalReward
from collections import defaultdict
from pipeline.utils_pipe import get_rhyme_id
from sacrebleu import corpus_bleu
# from sacrebleu.metrics import BLEU, CHRF, TER
# from torchmetrics.text import TranslationEditRate
from comet import download_model, load_from_checkpoint
model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)

# from bert_score import score
# from bert_score import BERTScorer
# hide the loading messages
# import logging
# import transformers
# transformers.tokenization_utils.logger.setLevel(logging.ERROR)
# transformers.configuration_utils.logger.setLevel(logging.ERROR)
# transformers.modeling_utils.logger.setLevel(logging.ERROR)
# scorer = BERTScorer(lang="zh")

# use_kimi = False #True
# N = 10

# if use_kimi:
#     client = OpenAI(
#         api_key="Y2w0dXFxMXI2a2plaXVudDFhdDA6bXNrLWF0RGxuUWllNjhmME9lZTJJcWtwYnRkbDE1bEo=",
#         base_url="https://api.moonshot.cn/v1")
# else:
#     client = OpenAI(
#         api_key="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",
#     )

'''新的test set'''
with open('data/musical_test.json', encoding="utf-8") as file_obj:
    dataset = json.load(file_obj)
'''之前的test set'''
# with open('final_test.json', encoding="utf-8") as file_obj:
#     dataset = json.load(file_obj)
# with open('test_set_reward3.json', encoding="utf-8") as file_obj:
#     dataset2 = json.load(file_obj)
#     dataset.extend(dataset2)
'''Ou的test set'''
# with open('data/parallel_test.json', encoding="utf-8") as file_obj:
#     dataset = json.load(file_obj)


data = defaultdict(lambda: defaultdict(list))
new_dataset = []
for row in dataset:
    en = row['en']
    zh = row['zh']
    trans = zh
    for ch in ',./，。、！!?？：:':
        trans = trans.replace(ch, '')
    length = len(trans)
    prompt = f'''I will give you a English lyric, and you need to translation it into Chinese with exactly {length} characters. Please only output the translated results and nothing more. The English lyrics is: {en}. Then the translation result is:'''
    if prompt in data:
        continue
    else:
        new_dataset.append(row)
        data[prompt] = row
with open('test_data123.json', 'w', encoding='utf-8') as file_obj:
    json.dump(new_dataset, file_obj, ensure_ascii=False)
dataset = new_dataset
print("gen new dataset finished", len(dataset))

model = EvalReward()
def cal(filename, tot_num=200000, only_bleu=False):
    if filename.endswith('.json'):
        with open(filename, encoding="utf-8") as file_obj:
            data = json.load(file_obj)
    else:
        with open(filename, encoding="utf-8") as file_obj:
            lines = file_obj.readlines()
            # print(lines)
        data = [line.strip() for line in lines if line.strip()]
        print(len(data), data[:5])
        
    tot_num = len(data)
    print('start', filename, tot_num)
    res1 = 0
    res2 = 0
    len_correct = 0
    len_err = 0
    scored_num = 0
    rhyme_correct = 0
    par_list = []
    zh_list = []
    en_list = []
    preds = []
    refs = []
    for idx in range(tot_num):
        par = dataset[idx]['par'] if 'par' in dataset[idx] else ''
        en = dataset[idx]['en']
        zh = data[idx]
        # r12 = model.eval_reward12(par, en, zh)
        # r3 = model.eval_reward3(par, en, zh)
        par_list.append(par)
        en_list.append(en)
        zh_list.append(zh)
        
        r_trans = zh.strip()
        d_trans = dataset[idx]['zh']
        for ch in ',./，。、！!?？：:':
            r_trans = r_trans.replace(ch, '')
            d_trans = d_trans.replace(ch, '')
        real_length = len(r_trans)
        desired_length = len(d_trans)
        len_correct += real_length == desired_length
        rhyme_correct += int(get_rhyme_id(d_trans) == get_rhyme_id(r_trans))
        preds.append(zh)
        refs.append(dataset[idx]['zh'])
        # res1 += r12
        # res2 += r3
        scored_num += 1
        # print("en=", en, "zh=", zh, "reward12=", r12, "reward3=", r3)
        # print("---------------------------", idx, " / ", tot_num, "finished-----------------------------------")
    bleu = round(corpus_bleu(preds, [refs], tokenize='zh').score, 4)
    # P, R, F1 = score(preds, refs, lang="zh", verbose=True)
    # P, R, F1 = scorer.score(preds, refs)
    data_for_comet = [{"src": en_list[i], "mt": preds[i], "ref": refs[i]} for i in range(len(refs))]
    comet = comet_model.predict(data_for_comet, batch_size=32, gpus=1)
    comet = round(comet.system_score, 4)


    # TER = TranslationEditRate()
    # ter = float(TER(preds, refs))
    if only_bleu:
        # print(tot_num, preds[:3], refs[:3], '\n', corpus_bleu(preds, [refs], tokenize='zh'))
        print(filename, "results are:", round(len_correct / scored_num, 3), round(rhyme_correct / scored_num, 3), bleu, comet)
        return
    res1 = model.eval_reward12_batch_samenum(par_list, en_list, zh_list)
    res2 = model.eval_reward3_batch_samenum(par_list, en_list, zh_list)
    res1 = sum(res1)
    res2 = sum(res2)
    print(filename, "results are:", round(len_correct / scored_num, 3), round(res1 / scored_num, 3), round(res2 / scored_num, 3), round(rhyme_correct / scored_num, 3), bleu, comet)

def cal_allres(filename):
    par_list = []
    zh_list = []
    en_list = []
    with open(filename, encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    for song in data:
        song_res = data[song]
        res_song = []
        for par_res in song_res:
            res_par = []
            for i, line_res in enumerate(par_res):
                par_list.append(line_res['par'])
                en_list.append(line_res['en_line'])
                zh_list.append(line_res['zh_trans'][0])
    print(len(par_list))
    model = EvalReward()
    res1 = model.eval_reward12_batch_samenum(par_list, en_list, zh_list)
    res2 = model.eval_reward3_batch_samenum(par_list, en_list, zh_list)
    print(sum(res1)/len(res1), sum(res2)/len(res2))

# cal('C_50w.json')
# cal('C_50w_worhyme.json')
# cal('testset_chinese_gt.json')
# cal('C_70w_filteredacc.json')
# cal('C_70w_filteredacc_worhyme.json')
# cal('C_100w.json')
# cal('C_100w_worhyme.json')
# cal('C_100w_filtered.json')
# cal('C_100w_filtered_worhyme.json')
# cal('C_100w_filtered2.json')
# cal('C_100w_filtered2_worhyme.json')

'''ou test set'''
# cal('ou_parallel_test.txt')
# cal('ou_testset_C_175w.json')
# cal('ou_testset_C_175w_worhyme.json')
# cal('ou_testset_C_175w_filtered.json')
# cal('ou_testset_C_175w_filtered_worhyme.json')
# cal('ou_testset_C_175w_filtered_accfintune.json')
# cal('ou_testset_C_175w_filtered_accfintune_worhyme.json')


'''our test set'''
# cal('ou_sentence_testres.txt')
# cal('C_175w.json')
# cal('C_175w_worhyme.json')
# cal('C_175w_filtered.json')
# cal('C_175w_filtered_worhyme.json')
# cal('C_175w_filtered_accfintune.json')
# cal('C_175w_filtered_accfintune_worhyme.json')
# cal('gpt4_worhyme.json')
cal('gpt4_worhyme_5shot.json')
# cal('gpt4_rhyme.json')
# cal('gpt4_rhyme_5shot.json')
# cal('kimichat.json')