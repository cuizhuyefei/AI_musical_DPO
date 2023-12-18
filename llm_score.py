import torch
torch.backends.cuda.matmul.allow_tf32 = True
import transformers
import re, os, json, random
from openai import OpenAI
from utils import (
    pad_to_length,
    get_local_dir,
)
from typing import Optional, Dict, List, Union, Tuple
from inference import get_batch_iterator

use_llama = True
use_kimi = False
N = 5

def get_batch_samples(model, tokenizer, batch: Dict[str, torch.LongTensor], n=1) -> Tuple[str, str]:
    """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

    # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
    # ctx = lambda: (FSDP.summon_full_params(model, writeback=False, recurse=False))
    # with ctx():
    reference_output = model.generate(
        batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], 
        max_length=500, do_sample=True, pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=n, temperature=1.0)
    # print(reference_output, reference_output.shape)
    reference_output = pad_to_length(reference_output, 500, tokenizer.pad_token_id)
    # print(reference_output, reference_output.shape)
    #reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
    reference_output_decoded = tokenizer.batch_decode(reference_output, skip_special_tokens=True)
    # print(reference_output_decoded)
    return reference_output_decoded

if use_llama:
    # loading tokenizer
    tokenizer_name_or_path = 'hfl/chinese-alpaca-2-13b'
    print(f'Loading tokenizer {tokenizer_name_or_path}')
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        tokenizer_name_or_path, 
        # cache_dir=get_local_dir(['.cache', '/scr-ssd', '/scr'])
    )
    # loading model
    policy_model_dtype = getattr(torch, 'float32')
    print(f'Loading model')
    policy_model3 = transformers.AutoModelForCausalLM.from_pretrained(
        'hfl/chinese-alpaca-2-13b', 
        cache_dir=get_local_dir(['.cache',]), 
        low_cpu_mem_usage=True, 
        torch_dtype=policy_model_dtype)
    
    state_dict = torch.load('.cache/zhuorui/reward3_2023-12-18_02-29-10_511656/LATEST/policy.pt', map_location='cpu')
    policy_model3.load_state_dict(state_dict['state'])
    device = 'cuda'
    policy_model3.to(device)
    policy_model3.eval() # set to eval mode!
elif use_kimi:
    client = OpenAI(
        api_key="Y2w0dXFxMXI2a2plaXVudDFhdDA6bXNrLWF0RGxuUWllNjhmME9lZTJJcWtwYnRkbDE1bEo=",
        base_url="https://api.moonshot.cn/v1")
else:
    client = OpenAI(
        api_key="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",
    )

with open('data_for_llm_scoring.json', encoding="utf-8") as file_obj:
    data = json.load(file_obj)

res_tot = []
idx = 0
random.seed(42)
random_list = random.sample(range(len(data)), 100)
# tot_zh = 0
# for idx, data_point in enumerate(data):
#     tot_zh += len(data_point['kimi']) #+len(data_point['llama'])
# print(tot_zh)
# exit(0)

def eval_llama(prompt, policy_model, N):
    data = [{
        'prompt': prompt,
        'responses': ['', ''],
        'pairs': [(0, 1)],
        'sft_target': '',
    }]
    # print('data', data)
    batch = get_batch_iterator(tokenizer, data)
    # print('batch', batch)
    reference_samples = get_batch_samples(policy_model, tokenizer, batch, n=N)
    response = [s[len(prompt):] for s in reference_samples]
    print(response[:5])
    return response
    
    # first_score = -1
    # mean_score = 0
    # num_score = 0
    # for s in reference_samples:
    #     score = [float(n) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', s[len(prompt):])]
    #     if len(score) > 0:
    #         mean_score += score[0]
    #         num_score += 1
    #         if num_score == 1:
    #             first_score = score[0]
    # mean_score /= num_score
    # return response, mean_score, first_score

for idx, data_point in enumerate(data):
    en = data_point['en']
    res = {}
    res['en'] = en
    res['zh'] = []
    res['scores'] = []
    res['llama_1'] = []
    res['pairs'] = []
    res['idx'] = idx
    if idx not in random_list:
        continue
    for zh in data_point['kimi']:
        prompt = f'''You are a translation grader. Given a Chinese translation of lyrics, you need to give scores in range 1-4 (4 is the highest) for whether it looks like good lyrics. 

Criteria for scoring:
Score 1: The translation does not resonate as good lyrics.
Score 2: Acceptable as lyrics, but mundane and unremarkable.
Score 3: Good fit for lyrics with some literary flair and aesthetic language.
Score 4: Outstanding lyrical quality, inventive, expressive, and captivating.

Reserve a score of 4 for truly impressive lyricism and be prudent when giving 4. Regular conversational phrases typically merit a score of 2.

Now, I will provide you with the Chinese translation. You need to give me only one number and nothing else. 
The Chinese translation is: {zh}. The score is:'''
        response3 = eval_llama(prompt, policy_model3, N) if use_llama else client.chat.completions.create(
            model="moonshot-v1-8k" if use_kimi else "gpt-4",
            messages=[{"role": "user", "content": prompt}],
            n=N,
            temperature=0.5
        )
        print(response3)
        continue
        try:
            prompt = f'''You are a Chinese sentence grader. Given Chinese sentence, you need to give scores in range 1-4 (4 is the highest) based on sentence fluency. 

Here are the metrics:
Score 1: Not fluent at all, can't understand the meaning easily.
Score 2: There are inappropriate or awkward phrases or other big unbearable flaws.
Score 3: Quite fluent. There exists small mistakes, but they are acceptable.
Score 4: Very fluent and no mistakes. Likely come from a native Chinese speaker.

Now, I will provide you with the Chinese sentence. You need to give me only one number and nothing else. 
The Chinese sentence is: {zh}'''
            
            response1 = eval_llama(prompt, policy_model1, N) if use_llama else client.chat.completions.create(
                model="moonshot-v1-8k" if use_kimi else "gpt-4",
                messages=[{"role": "user", "content": prompt}],
                n=N,
                temperature=0.5
            )
        except:
            continue
        try:
            prompt = f'''You are a translation grader. Given English lyrics and a corresponding Chinese translation, you need to give scores in range 1-4 (4 is the highest) for translation accuracy. 

Here are the metrics for translation accuracy:
Score 1: More than 50% is translated wrongly, or there are unbearable translation mistakes.
Score 2: Merely acceptable, but there are mistakes that need correction.
Score 3: No big mistake in translation, totally acceptable. But there is still space for improvement, such as phrasing or the translation of idioms.
Score 4: Excellent translation.

Note that in either metrics, score 4 means excellent and should be only given if you are absolutely sure the translated sentence is perfect. Any tiny mistake will make its score less than 4.

Now, I will provide you with the English lyrics and the Chinese translation. You need to give me only one number and nothing else. 
The English lyrics is: {en}.
The Chinese translation is: {zh}. The score is:'''
            response2 = eval_llama(prompt, policy_model2, N) if use_llama else client.chat.completions.create(
                model="moonshot-v1-8k" if use_kimi else "gpt-4",
                messages=[{"role": "user", "content": prompt}],
                n=N,
                temperature=0.5
            )
        except:
            continue
        try:
            prompt = f'''You are a translation grader. Given a Chinese translation of lyrics, you need to give scores in range 1-4 (4 is the highest) for whether it looks like good lyrics. 

Criteria for scoring:
Score 1: The translation does not resonate as good lyrics.
Score 2: Acceptable as lyrics, but mundane and unremarkable.
Score 3: Good fit for lyrics with some literary flair and aesthetic language.
Score 4: Outstanding lyrical quality, inventive, expressive, and captivating.

Reserve a score of 4 for truly impressive lyricism and be prudent when giving 4. Regular conversational phrases typically merit a score of 2.

Now, I will provide you with the Chinese translation. You need to give me only one number and nothing else. 
The Chinese translation is: {zh}. The score is:'''
            response3 = eval_llama(prompt, policy_model3, N) if use_llama else client.chat.completions.create(
                model="moonshot-v1-8k" if use_kimi else "gpt-4",
                messages=[{"role": "user", "content": prompt}],
                n=N,
                temperature=0.5
            )
        except:
            continue
        r1 = 0
        r2 = 0
        r3 = 0
        for i in range(N):
            score1 = [float(n) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', response1.choices[i].message.content)][0]
            score2 = [float(n) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', response2.choices[i].message.content)][0]
            score3 = [float(n) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', response2.choices[i].message.content)][0]
            r1 += score1
            r2 += score2
            r3 += score3
        r1 /= N
        r2 /= N
        r3 /= N
        print("idx=", idx, "en=", en, "zh=", zh, "fluency=", r1, "translation=", r2, "quality=", r3)
        res['zh'].append(zh)
        res['scores'].append((r1, r2, r3))
        if len(res['zh'])>=3: break # gpt-4 only give 200
    # num = len(res['zh'])
    # for i in range(num):
    #     for j in range(i):
    #         d1 = res['scores'][i][0] - res['scores'][j][0]
    #         d2 = res['scores'][i][1] - res['scores'][j][1]
    #         if d1 > 0.5 and d2 > 0.5:
    #             res["pairs"].append((i, j))
    #         elif d1 < -0.5 and d2 < -0.5:
    #             res["pairs"].append((j, i))
    #         elif d1 > 0 and d2 > 1.0:
    #             res["pairs"].append((i, j))
    #         elif d1 < 0 and d2 < -1.0:
    #             res["pairs"].append((j, i))
    #         elif d1 > 1.0 and d2 > 0:
    #             res["pairs"].append((i, j))
    #         elif d1 < -1.0 and d2 < 0:
    #             res["pairs"].append((j, i))
    res_tot.append(res)
    with open('llama_scores_kimi_data.json', 'w', encoding='utf-8') as file_obj:
        json.dump(res_tot, file_obj, ensure_ascii=False)
    idx += 1
    print("---------------------------", idx, "finished-----------------------------------")

print(res_tot)    


# data = [
#     {
#         'en': "I have never been satisfied.", 
#         'zh': ["我从未满足过自己的", "我从来都不曾满足了", "我从来都不曾满足过", "我从来都不满足过去", "我从来都不会满足了", "我从未满足，始终如此"],
#     },
#     {
#         'en': 'The sun is shining,',
#         'zh': ['太阳在闪', '阳光灿烂'],
#     },
#     {
#         'en': 'When you are sixteen going on seventeen.',
#         'zh': ["当你六十一岁去年满十七", "当你已经六十一岁而未满", "当你年纪是十六岁又一岁", "当你十六岁快要十七岁时", "当你十六岁，迈向十七岁轮", "当你十六岁，迈向十七岁"],
#     },
#     {
#         'en': 'How do you keep a wave upon the sand?',
#         'zh': ["任何地方你去我跟去", "你如何保持沙滩的浪花", "你是如何在沙滩上守望", "你怎么能把海浪留在沙", "你怎么能让浪花停在沙", "你如何在沙滩上留下波", "如何让沙上的浪不消散", "如何让沙上的波浪不散", "如何让海浪停留在沙滩"],
#     }
# ]