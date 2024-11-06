import torch
torch.backends.cuda.matmul.allow_tf32 = True
import transformers
import re, os, json, random
import tqdm
from openai import OpenAI
from utils import (
    pad_to_length,
    get_local_dir,
    gpu_memory_usage,
)
from typing import Optional, Dict, List, Union, Tuple
from inference import get_batch_iterator
from cal_corelation import get_corelation
import numpy as np
call_inference_cnt_for_empty = 0
 
 # 加一个top_p=0.95试试！
 # reward_model: no randomness; otherwise use sampling
def get_batch_samples(model, tokenizer, batch: Dict[str, torch.LongTensor], n=1, T=0.2, reward_model=False) -> Tuple[str, str]:
    """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

    # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
    # ctx = lambda: (FSDP.summon_full_params(model, writeback=False, recurse=False))
    # with ctx():
    if n >= 80:
        return get_batch_samples(model, tokenizer, batch, n//2, T, reward_model) + get_batch_samples(model, tokenizer, batch, n//2, T, reward_model)
    global call_inference_cnt_for_empty
    call_inference_cnt_for_empty += 1
    if call_inference_cnt_for_empty % 10 == 0 or not reward_model:
        torch.cuda.empty_cache()
    device = next(model.parameters()).device
    max_length = 500 if reward_model else 120 # reward model / generation
    # beam_search: , num_beams=5
    if not reward_model:
        reference_output = model.generate(
            batch['prompt_input_ids'].to(device), attention_mask=batch['prompt_attention_mask'].to(device), 
            max_length=max_length, do_sample=True, pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=n, temperature=T, top_p=0.95) #, temperature=T, top_p=0.95
    else:
        assert n==1
        reference_output = model.generate(
            batch['prompt_input_ids'].to(device), attention_mask=batch['prompt_attention_mask'].to(device), 
            max_length=max_length, do_sample=False, pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=n)
    reference_output = pad_to_length(reference_output, max_length, tokenizer.pad_token_id)
    # if n>20:
    #     print(gpu_memory_usage())
    #reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
    reference_output_decoded = tokenizer.batch_decode(reference_output, skip_special_tokens=True)
    return reference_output_decoded

# deterministic by default for reward model
# T=1 for llamaGenAPI
def eval_llama(prompts, policy_model, tokenizer, N=1, use_FSDP=False, T=0.001, reward_model=False):
    assert use_FSDP==False
    data = []
    idx = 0
    ret = []

    for prompt in tqdm.tqdm(prompts, desc='evaluation'):
        data.append({
            'prompt': prompt,
            'responses': ['', ''],
            'pairs': [(0, 1)],
            'sft_target': '',
        })
        idx += 1
        if idx % 16 == 0:
            batch = get_batch_iterator(tokenizer, data, False)
            reference_samples = get_batch_samples(policy_model, tokenizer, batch, n=N, T=T, reward_model=reward_model)
            for jdx, prompt in enumerate(batch['prompt']):
                for i in range(jdx * N, jdx * N + N):
                    trans = reference_samples[i][len(prompt):].strip()
                    ret.append(trans)
            data = []

    if len(data) > 0:
        batch = get_batch_iterator(tokenizer, data, False)
        # print("batch is already generated!", idx)
        reference_samples = get_batch_samples(policy_model, tokenizer, batch, n=N, T=T, reward_model=reward_model)
        for jdx, prompt in enumerate(batch['prompt']):
            for i in range(jdx * N, jdx * N + N):
                trans = reference_samples[i][len(prompt):].strip()
                ret.append(trans)
        data = []
    if len(ret) == 1:
        return ret[0]
    return ret
    
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

# def eval_gpt(prompt, N):
#     response = client.chat.completions.create(
#         model="moonshot-v1-8k" if use_kimi else "gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}],
#         n=N,
#         temperature=0.2
#     )
#     return [response.choices[i].message.content for i in range(N)]

# def get_batch_samples_FSDP(model, tokenizer, batch: Dict[str, torch.LongTensor], n=1, T=0.2) -> Tuple[str, str]:
#     """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

#     # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
#     import torch.distributed as dist
#     from torch.distributed.fsdp import (
#         FullyShardedDataParallel as FSDP,
#         MixedPrecision,
#         StateDictType,
#         BackwardPrefetch,
#         ShardingStrategy,
#         CPUOffload,
#     )
#     ctx = lambda: (FSDP.summon_full_params(model, writeback=False, recurse=False))
#     device = next(model.parameters()).device
#     from utils import gpu_memory_usage
#     # print('device', device, gpu_memory_usage())
#     with ctx():
#         reference_output = model.generate(
#             batch['prompt_input_ids'].to(device), attention_mask=batch['prompt_attention_mask'].to(device), 
#             max_length=500, do_sample=True, pad_token_id=tokenizer.pad_token_id,
#             num_return_sequences=n, temperature=T)
#     # print('device', device, gpu_memory_usage())
#     reference_output = pad_to_length(reference_output, 500, tokenizer.pad_token_id)
#     #reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
#     reference_output_decoded = tokenizer.batch_decode(reference_output, skip_special_tokens=True)
#     return reference_output_decoded

def evalReward2(model, tokenizer):
    print('enter loading')
    with open('final_test.json', encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    print('finish loading', len(data))
    res = []
    for idx, data_point in enumerate(data):
        en = data_point['en']
        zh = data_point['zh']
        par = data_point['par']
        
        prompt = f'''You are a translation grader. Given English lyrics and a corresponding Chinese translation, you need to give scores in range 1-4 (4 is the highest) for translation accuracy. 

Here are the metrics for translation accuracy:
Score 1: More than 50% is translated wrongly, or there are unbearable translation mistakes.
Score 2: Merely acceptable, but there are mistakes that need correction.
Score 3: No big mistake in translation, totally acceptable. But there is still space for improvement, such as phrasing or the translation of idioms.
Score 4: Excellent translation.

Note that in either metrics, score 4 means excellent and should be only given if you are absolutely sure the translated sentence is perfect. Any tiny mistake will make its score less than 4.

Now, I will provide you with the English lyrics and the Chinese translation. You need to give me only one number and nothing else. 
For a comprehensive understanding, the context is: {par}.
The English lyrics is: {en}.
The Chinese translation is: {zh}. The score is: '''
        response = eval_llama([prompt], model, tokenizer, 1, use_FSDP=True)[0]
        print(response)
        reward2 = [float(n) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', response)][0]
        print(reward2)

        res.append((idx, en, zh, reward2, data_point['human_scores'][1]))
        print(idx, en, zh, reward2, data_point['human_scores'][1])
        idx += 1
    correlation = np.corrcoef([data[3] for data in res], [data[4] for data in res])[0, 1]
    return res, correlation

class llamaGenAPI:
    def __init__(self, ckpt=3):
        # loading tokenizer
        tokenizer_name_or_path = 'hfl/chinese-alpaca-2-13b'
        print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(
            tokenizer_name_or_path, 
            # cache_dir=get_local_dir(['.cache', '/scr-ssd', '/scr'])
        )
        # loading model
        policy_model_dtype = getattr(torch, 'float32')
        print(f'Loading model')
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            'hfl/chinese-alpaca-2-13b', 
            cache_dir=get_local_dir(['.cache',]), 
            low_cpu_mem_usage=True, 
            torch_dtype=policy_model_dtype)
        if ckpt > 0:
            if ckpt==3:
                state_dict = torch.load('.cache/zhuorui/C_filtered_sft_2024-02-07_19-29-40_317953/LATEST/policy.pt', map_location='cpu')
            elif ckpt==2:
                state_dict = torch.load('.cache/zhuorui/C_filtered_sft_2024-02-06_08-15-59_499366/LATEST/policy.pt', map_location='cpu')
            elif ckpt==1:
                state_dict = torch.load('.cache/zhuorui/C_sft_2024-02-08_07-51-21_911116/LATEST/policy.pt', map_location='cpu')
            else:
                assert False
            # .cache/zhuorui/C_sft_2024-02-08_07-51-21_911116/LATEST/policy.pt (175w C)
            # .cache/zhuorui/C_filtered_sft_2024-02-06_08-15-59_499366/LATEST/policy.pt (175w filtered C)
            # .cache/zhuorui/C_filtered_sft_2024-02-07_19-29-40_317953/LATEST/policy.pt (175w filtered + 70w acc finetune)
            self.model.load_state_dict(state_dict['state'])
        else:
            print('[NOTE] NO FINETUNE')
        device = 'cuda:2'
        self.model.to(device)
        self.model.eval() # set to eval mode!

    def call(self, prompt, N):
        return eval_llama([prompt], self.model, self.tokenizer, N, T=0.7)

class EvalReward:
    def __init__(self):
        # loading tokenizer
        tokenizer_name_or_path = 'hfl/chinese-alpaca-2-13b'
        print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(
            tokenizer_name_or_path, 
            # cache_dir=get_local_dir(['.cache', '/scr-ssd', '/scr'])
        )
        # loading model
        policy_model_dtype = getattr(torch, 'float16')
        print(f'Loading model')
        # from transformers import LlamaConfig,LlamaForCausalLM,LlamaTokenizer
        # from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model
        # from accelerate import load_checkpoint_and_dispatch
        # device_map = infer_auto_device_map(self.policy_model2, max_memory = {int(cuda):'50GiB' for cuda in '0,1,2,3,4,5,6,7'.split(',')} ,no_split_module_classes=LlamaForCausalLM._no_split_modules) #自动划分每个层的设备
        # state_dict = torch.load('.cache/zhuorui/reward_ft_6_2024-01-03_22-36-12_077105/LATEST/policy.pt', map_location='cpu')
        # self.policy_model2.load_state_dict(state_dict['state'])
        # self.policy_model2 = dispatch_model(self.policy_model2,device_map=device_map) #并分配到具体的设备上

        print('loading reward model', 'cuizhuyefei/rewardbasic')
        self.policy_model2 = transformers.AutoModelForCausalLM.from_pretrained(
            '/usr0/home/zhuoruiy/backup/rewardbasic_fp16', 
            cache_dir=get_local_dir(['.cache',]), 
            low_cpu_mem_usage=True, 
            torch_dtype=policy_model_dtype)
        device = 'cuda:0'
        self.policy_model2.to(device)
        self.policy_model2.eval() # set to eval mode!
        self.policy_model3 = transformers.AutoModelForCausalLM.from_pretrained(
            '/usr0/home/zhuoruiy/backup/rewardadvanced_fp16', 
            cache_dir=get_local_dir(['.cache',]), 
            low_cpu_mem_usage=True, 
            torch_dtype=policy_model_dtype)
        device = 'cuda:1'
        self.policy_model3.to(device)
        self.policy_model3.eval() # set to eval mode!
    
    def eval_reward12_batch_samenum(self, par_list, en_list, zh_list):
        # self.policy_model2.to('cuda')
        n = len(par_list)
        assert(len(en_list) == n and len(zh_list) == n)
        prompts = []
        for idx in range(n):
            en = en_list[idx]
            zh = zh_list[idx]
            par = par_list[idx]
            prompt = f'''You are a translation grader. Given English lyrics and a corresponding Chinese translation, you need to give scores in range 1-4 (4 is the highest) considering both fluency and translation accuracy. 

Here are the metrics:
Score 1: Not very fluent. There are inappropriate or awkward phrases ot other big flaws.
Score 2: Quite fluent, but there are serious translation mistakes that need correction.
Score 3: Quite fluent, no big mistake in translation. But there are still small mistakes in phrasing or the translation of idioms.
Score 4: Very fluent, no mistake, and excellent translation.

Note that score 4 means excellent and should be only given if you are absolutely sure the translated sentence is perfect. Any tiny mistake will make its score less than 4.

Now, I will provide you with the English lyrics and the Chinese translation. You need to give me only one number and nothing else. 
For a comprehensive understanding, I will provide you the context: {par}.
The English lyrics is: {en}.
The Chinese translation is: {zh}. The score is: '''
            prompts.append(prompt)
        responses = eval_llama(prompts, self.policy_model2, self.tokenizer, 1, reward_model=True)
        ret = []
        for idx in range(n):
            nums = [int(float(n)) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', responses[idx])]
            if len(nums) == 0:
                ret.append(0)
            else:
                ret.append(nums[0])
        # self.policy_model2.to('cpu')
        return ret
    
    def eval_reward3_batch_samenum(self, par_list, en_list, zh_list):
        # self.policy_model3.to('cuda')
        n = len(par_list)
        assert(len(en_list) == n and len(zh_list) == n)
        prompts = []
        for idx in range(n):
            en = en_list[idx]
            zh = zh_list[idx]
            par = par_list[idx]
            prompt = f'''You are a translation grader. Given a Chinese translation of lyrics, you need to give scores in range 1-4 (4 is the highest) for whether it looks like good lyrics. 

Criteria for scoring:
Score 1: The translation does not resonate as good lyrics.
Score 2: Acceptable as lyrics, but mundane and unremarkable.
Score 3: Good fit for lyrics with some literary flair and aesthetic language.
Score 4: Outstanding lyrical quality, inventive, expressive, and captivating.

Reserve a score of 4 for truly impressive lyricism and be prudent when giving 4. Regular conversational phrases typically merit a score of 2.

Now, I will provide you with the Chinese translation. You need to give me only one number and nothing else. 
The Chinese translation is: {zh}. The score is'''
            prompts.append(prompt)
        responses = eval_llama(prompts, self.policy_model3, self.tokenizer, 1, reward_model=True)
        ret = []
        for idx in range(n):
            nums = [int(float(n)) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', responses[idx])]
            if len(nums) == 0:
                ret.append(0)
            else:
                ret.append(nums[0])
        # self.policy_model3.to('cpu')
        return ret
    
    def eval_reward12_batch(self, par, en_list, zh_lists):
        # self.policy_model2.to('cuda')
        prompts = []
        for idx, en in enumerate(en_list):
            for zh in zh_lists[idx]:
                if len(zh) > 18:
                    zh = zh[:18]
                prompt = f'''You are a translation grader. Given English lyrics and a corresponding Chinese translation, you need to give scores in range 1-4 (4 is the highest) considering both fluency and translation accuracy. 

Here are the metrics:
Score 1: Not very fluent. There are inappropriate or awkward phrases ot other big flaws.
Score 2: Quite fluent, but there are serious translation mistakes that need correction.
Score 3: Quite fluent, no big mistake in translation. But there are still small mistakes in phrasing or the translation of idioms.
Score 4: Very fluent, no mistake, and excellent translation.

Note that score 4 means excellent and should be only given if you are absolutely sure the translated sentence is perfect. Any tiny mistake will make its score less than 4.

Now, I will provide you with the English lyrics and the Chinese translation. You need to give me only one number and nothing else. 
For a comprehensive understanding, I will provide you the context: {par}.
The English lyrics is: {en}.
The Chinese translation is: {zh}. The score is: '''
                prompts.append(prompt)
        m = int(len(prompts) / len(en_list))
        responses = eval_llama(prompts, self.policy_model2, self.tokenizer, 1, reward_model=True)
        ret = []
        for i in range(0, len(prompts), m):
            cur_ret = []
            for j in range(i, i + m):
                nums = [int(float(n)) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', responses[j])]
                if len(nums) == 0:
                    cur_ret.append(0)
                else:
                    cur_ret.append(nums[0])
            ret.append(cur_ret)
        # ret = [[[int(float(n)) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', responses[j])][0] for j in range(i, i + m)] for i in range(0, len(prompts), m)]
        # print("ret=", ret)
        # self.policy_model2.to('cpu')
        return ret
    
    def eval_reward12(self, par, en, zh):
        if len(zh) > 18:
            zh = zh[:18]
        prompt = f'''You are a translation grader. Given English lyrics and a corresponding Chinese translation, you need to give scores in range 1-4 (4 is the highest) considering both fluency and translation accuracy. 

Here are the metrics:
Score 1: Not very fluent. There are inappropriate or awkward phrases ot other big flaws.
Score 2: Quite fluent, but there are serious translation mistakes that need correction.
Score 3: Quite fluent, no big mistake in translation. But there are still small mistakes in phrasing or the translation of idioms.
Score 4: Very fluent, no mistake, and excellent translation.

Note that score 4 means excellent and should be only given if you are absolutely sure the translated sentence is perfect. Any tiny mistake will make its score less than 4.

Now, I will provide you with the English lyrics and the Chinese translation. You need to give me only one number and nothing else. 
For a comprehensive understanding, I will provide you the context: {par}.
The English lyrics is: {en}.
The Chinese translation is: {zh}. The score is: '''
        return int(eval_llama([prompt], self.policy_model2, self.tokenizer, 1, reward_model=True)[0])
    
    def eval_reward3_batch(self, par, en_list, zh_lists):
        # self.policy_model3.to('cuda')
        prompts = []
        for idx, en in enumerate(en_list):
            for zh in zh_lists[idx]:
                if len(zh) > 18:
                    zh = zh[:18]
                prompt = f'''You are a translation grader. Given a Chinese translation of lyrics, you need to give scores in range 1-4 (4 is the highest) for whether it looks like good lyrics. 

    Criteria for scoring:
    Score 1: The translation does not resonate as good lyrics.
    Score 2: Acceptable as lyrics, but mundane and unremarkable.
    Score 3: Good fit for lyrics with some literary flair and aesthetic language.
    Score 4: Outstanding lyrical quality, inventive, expressive, and captivating.

    Reserve a score of 4 for truly impressive lyricism and be prudent when giving 4. Regular conversational phrases typically merit a score of 2.

    Now, I will provide you with the Chinese translation. You need to give me only one number and nothing else. 
    The Chinese translation is: {zh}. The score is'''
                prompts.append(prompt)
        m = int(len(prompts) / len(en_list))
        responses = eval_llama(prompts, self.policy_model3, self.tokenizer, 1, reward_model=True)
        ret = []
        for i in range(0, len(prompts), m):
            cur_ret = []
            for j in range(i, i + m):
                nums = [int(float(n)) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', responses[j])]
                if len(nums) == 0:
                    cur_ret.append(0)
                else:
                    cur_ret.append(nums[0])
            ret.append(cur_ret)
        # ret = [[[int(float(n)) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', responses[j])][0] for j in range(i, i + m)] for i in range(0, len(prompts), m)]
        # print("ret=", ret)
        # self.policy_model3.to('cpu')
        return ret
    
    def eval_reward3(self, par, en, zh):
        if len(zh) > 18:
            zh = zh[:18]
        prompt = f'''You are a translation grader. Given a Chinese translation of lyrics, you need to give scores in range 1-4 (4 is the highest) for whether it looks like good lyrics. 

    Criteria for scoring:
    Score 1: The translation does not resonate as good lyrics.
    Score 2: Acceptable as lyrics, but mundane and unremarkable.
    Score 3: Good fit for lyrics with some literary flair and aesthetic language.
    Score 4: Outstanding lyrical quality, inventive, expressive, and captivating.

    Reserve a score of 4 for truly impressive lyricism and be prudent when giving 4. Regular conversational phrases typically merit a score of 2.

    Now, I will provide you with the Chinese translation. You need to give me only one number and nothing else. 
    The Chinese translation is: {zh}. The score is:'''
        return int(eval_llama([prompt], self.policy_model3, self.tokenizer, 1, reward_model=True)[0])



class Seq2SeqRefiner: # NLP project
    def __init__(self, nofinetune=False):
        # loading tokenizer
        tokenizer_name_or_path = 'hfl/chinese-alpaca-2-13b'
        print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(
            tokenizer_name_or_path, 
            # cache_dir=get_local_dir(['.cache', '/scr-ssd', '/scr'])
        )
        # loading model
        policy_model_dtype = getattr(torch, 'float32')
        print(f'Loading model')
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            'hfl/chinese-alpaca-2-13b', 
            cache_dir=get_local_dir(['.cache',]), 
            low_cpu_mem_usage=True, 
            torch_dtype=policy_model_dtype)
        if nofinetune == False:
            state_dict = torch.load('.cache/zhuorui/prompt_refine_2024-01-15_19-43-45_463291/LATEST/policy.pt', map_location='cpu')
            self.model.load_state_dict(state_dict['state'])
        device = 'cuda:2'
        self.model.to(device)
        self.model.eval() # set to eval mode!

    def call(self, prompt):
        return eval_llama([prompt], self.model, self.tokenizer, 1, T=0.2)


def draw_ft():
    use_llama = True
    use_kimi = False
    N = 5
    human_list = [[], [], []]
    llama_list = [[], [], []]
    llama_list_ft = []
    human_list_ft = []
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
        if False:
            policy_model1 = transformers.AutoModelForCausalLM.from_pretrained(
                'hfl/chinese-alpaca-2-13b', 
                cache_dir=get_local_dir(['.cache',]), 
                low_cpu_mem_usage=True, 
                torch_dtype=policy_model_dtype)    
            state_dict = torch.load('.cache/zhuorui/reward1_2023-12-18_02-19-56_212480/LATEST/policy.pt', map_location='cpu')
            policy_model1.load_state_dict(state_dict['state'])
            device = 'cuda:1'
            policy_model1.to(device)
            policy_model1.eval() # set to eval mode!
            policy_model2 = transformers.AutoModelForCausalLM.from_pretrained(
                'hfl/chinese-alpaca-2-13b', 
                cache_dir=get_local_dir(['.cache',]), 
                low_cpu_mem_usage=True, 
                torch_dtype=policy_model_dtype)
        
            # state_dict = torch.load('.cache/zhuorui/reward2_lr1e-5_2023-12-18_23-31-26_482219/LATEST/policy.pt', map_location='cpu')
            # state_dict = torch.load('.cache/zhuorui/reward2_2023-12-20_05-21-30_461036/LATEST/policy.pt', map_location='cpu')
            state_dict = torch.load('.cache/zhuorui/reward2_adjust_2024-01-02_23-57-49_409988/LATEST/policy.pt', map_location='cpu')
            policy_model2.load_state_dict(state_dict['state'])
            device = 'cuda:2'
            policy_model2.to(device)
            policy_model2.eval() # set to eval mode!
            policy_model3 = transformers.AutoModelForCausalLM.from_pretrained(
                'hfl/chinese-alpaca-2-13b', 
                cache_dir=get_local_dir(['.cache',]), 
                low_cpu_mem_usage=True, 
                torch_dtype=policy_model_dtype)
            
            state_dict = torch.load('.cache/zhuorui/reward3_2023-12-18_02-29-10_511656/LATEST/policy.pt', map_location='cpu')
            policy_model3.load_state_dict(state_dict['state'])
            device = 'cuda:3'
            policy_model3.to(device)
            policy_model3.eval() # set to eval mode!
        policy_model = transformers.AutoModelForCausalLM.from_pretrained(
                'hfl/chinese-alpaca-2-13b', 
                cache_dir=get_local_dir(['.cache',]), 
                low_cpu_mem_usage=True, 
                torch_dtype=policy_model_dtype)
        # state_dict = torch.load('.cache/zhuorui/reward_ft_2024-01-03_03-10-25_434623/LATEST/policy.pt', map_location='cpu')
        # state_dict = torch.load('.cache/zhuorui/reward_ft_2_2024-01-03_08-02-55_449614/LATEST/policy.pt', map_location='cpu')
        # state_dict = torch.load('.cache/zhuorui/reward_ft_3_2024-01-03_19-21-05_476711/LATEST/policy.pt', map_location='cpu')
        # state_dict = torch.load('.cache/zhuorui/reward_ft_4_2024-01-03_20-12-02_965985/LATEST/policy.pt', map_location='cpu')
        # state_dict = torch.load('.cache/zhuorui/reward_ft_5_2024-01-03_21-15-32_537539/LATEST/policy.pt', map_location='cpu')
        state_dict = torch.load('.cache/zhuorui/reward_ft_6_2024-01-03_22-36-12_077105/LATEST/policy.pt', map_location='cpu')
        # state_dict = torch.load('.cache/zhuorui/reward_ft_7_2024-01-03_23-38-10_632328/LATEST/policy.pt', map_location='cpu')
        # state_dict = torch.load('.cache/zhuorui/reward_ft_9_2024-01-04_03-52-27_468707/LATEST/policy.pt', map_location='cpu')
        # state_dict = torch.load('.cache/zhuorui/reward_ft_10_2024-01-04_07-03-12_089021/LATEST/policy.pt', map_location='cpu')
        
        policy_model.load_state_dict(state_dict['state'])
        device = 'cuda:1'
        policy_model.to(device)
        policy_model.eval() # set to eval mode!
    elif use_kimi:
        client = OpenAI(
            api_key="Y2w0dXFxMXI2a2plaXVudDFhdDA6bXNrLWF0RGxuUWllNjhmME9lZTJJcWtwYnRkbDE1bEo=",
            base_url="https://api.moonshot.cn/v1")
    else:
        client = OpenAI(
            api_key="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",
        )

    # with open('data_for_llm_scoring.json', encoding="utf-8") as file_obj:
    with open('final_test.json', encoding="utf-8") as file_obj:
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


    for idx, data_point in enumerate(data):
        en = data_point['en']
        par = data_point['par']
        res = {}
        res['en'] = en
        res['zh'] = []
        res['scores'] = []
        res['llama_1'] = []
        res['pairs'] = []
        res['idx'] = idx
        # if idx not in random_list:
        #     continue
        # for zh in data_point['kimi']:
        zh = data_point['zh']
        if False:
            try:
                prompt = f'''You are a Chinese sentence grader. Given Chinese sentence, you need to give scores in range 1-4 (4 is the highest) based on sentence fluency. 

    Here are the metrics:
    Score 1: Not fluent at all, can't understand the meaning easily.
    Score 2: There are inappropriate or awkward phrases or other big unbearable flaws.
    Score 3: Quite fluent. There exists small mistakes, but they are acceptable.
    Score 4: Very fluent and no mistakes. Likely come from a native Chinese speaker.

    Now, I will provide you with the Chinese sentence. You need to give me only one number and nothing else. 
    The Chinese sentence is: {zh}. The score is:'''
                response1 = eval_llama([prompt], policy_model1, tokenizer, N) if use_llama else eval_gpt(prompt, N)
                # print(response1)
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
    For a comprehensive understanding, the context is: {par}.
    The English lyrics is: {en}.
    The Chinese translation is: {zh}. The score is: '''
                response2 = eval_llama([prompt], policy_model2, tokenizer, N) if use_llama else eval_gpt(prompt, N)
                # print(response2)
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
                response3 = eval_llama([prompt], policy_model3, tokenizer, N) if use_llama else eval_gpt(prompt, N)
                # print(response3)
            except:
                continue
        else:
            try:
                prompt = f'''You are a translation grader. Given English lyrics and a corresponding Chinese translation, you need to give scores in range 1-4 (4 is the highest) considering both fluency and translation accuracy. 

Here are the metrics:
Score 1: Not very fluent. There are inappropriate or awkward phrases ot other big flaws.
Score 2: Quite fluent, but there are serious translation mistakes that need correction.
Score 3: Quite fluent, no big mistake in translation. But there are still small mistakes in phrasing or the translation of idioms.
Score 4: Very fluent, no mistake, and excellent translation.

Note that score 4 means excellent and should be only given if you are absolutely sure the translated sentence is perfect. Any tiny mistake will make its score less than 4.

Now, I will provide you with the English lyrics and the Chinese translation. You need to give me only one number and nothing else. 
For a comprehensive understanding, I will provide you the context: {par}.
The English lyrics is: {en}.
The Chinese translation is: {zh}. The score is: '''
                response = eval_llama([prompt], policy_model, tokenizer, N) if use_llama else eval_gpt(prompt, N)
            except:
                continue
        if False:
            r1 = 0
            r2 = 0
            r3 = 0
            for i in range(N):
                score1 = [float(n) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', response1[i])][0]
                try:
                    score2 = [float(n) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', response2[i])][0]
                except:
                    score2 = 1
                score3 = [float(n) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', response3[i])][0]
                r1 += score1
                r2 += score2
                r3 += score3
            r1 /= N
            r2 /= N
            r3 /= N
            print("idx=", idx, "en=", en, "zh=", zh, "fluency=", r1, "translation=", r2, "quality=", r3, "human=", data_point['human_scores'])
            res['zh'].append(zh)
            res['scores'].append((r1, r2, r3))
            if len(res['zh'])>=3: break # gpt-4 only give 200
            
            rs = [r1, r2, r3]
            for jdx in range(3):
                human_list[jdx].append(data_point['human_scores'][jdx])
                llama_list[jdx].append(int(round(rs[jdx])))
        else:
            r = 0
            for i in range(N):
                score = [float(n) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', response[i])][0]
                r += score
            r /= N
            
            llama_list_ft.append(int(round(r)))
            human_score = 0
            if data_point['human_scores'][0] <= 2: human_score = 1
            elif data_point['human_scores'][1] <=2 : human_score = 2
            elif data_point['human_scores'][0] == 4 and data_point['human_scores'][1] == 4: human_score = 4
            else: human_score = 3
            human_list_ft.append(human_score)
            print("idx=", idx, "en=", en, "zh=", zh, "ft score=", r, int(round(r)), 'human score=', human_score)
        
        idx += 1
        print("---------------------------", idx, "finished-----------------------------------")

    # print(res_tot)   
    print("begin to get corelation") 
    if False:
        get_corelation(human_list[0], llama_list[0], 'human vs llama scores on reward1', 'human scores', 'llama scores', 'human_llama_reward1.png')
        get_corelation(human_list[1], llama_list[1], 'human vs llama scores on reward2', 'human scores', 'llama scores', 'human_llama_reward2.png')
        get_corelation(human_list[2], llama_list[2], 'human vs llama scores on reward3', 'human scores', 'llama scores', 'human_llama_reward3.png')
    else:
        get_corelation(human_list_ft, llama_list_ft, 'human vs llama on reward_ft', 'human scores', 'llama scores', 'human_llama_reward_ft.png')
        # get_corelation(human_list_ft, llama_list_ft, 'human vs kimi on reward_ft', 'human scores', 'kimi scores', 'human_kimi_reward_ft.png')
        with open('llama_test_scores.json', 'w', encoding='utf-8') as file_obj:
            json.dump(llama_list_ft, file_obj, ensure_ascii=False)
        with open('human_test_scores.json', 'w', encoding='utf-8') as file_obj:
            json.dump(human_list_ft, file_obj, ensure_ascii=False)


def draw_reward3():
    use_llama = True
    use_kimi = False
    N = 5
    human_list = [[], [], []]
    llama_list = [[], [], []]
    llama_list_reward3 = []
    human_list_reward3 = []
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
        policy_model = transformers.AutoModelForCausalLM.from_pretrained(
                'hfl/chinese-alpaca-2-13b', 
                cache_dir=get_local_dir(['.cache',]), 
                low_cpu_mem_usage=True, 
                torch_dtype=policy_model_dtype)
        # state_dict = torch.load('.cache/zhuorui/reward3_1_2024-01-07_04-55-56_971086/LATEST/policy.pt', map_location='cpu')
        state_dict = torch.load('.cache/zhuorui/reward3_2_2024-01-07_05-23-50_091251/LATEST/policy.pt', map_location='cpu')
        policy_model.load_state_dict(state_dict['state'])
        device = 'cuda:1'
        policy_model.to(device)
        policy_model.eval() # set to eval mode!
    elif use_kimi:
        client = OpenAI(
            api_key="Y2w0dXFxMXI2a2plaXVudDFhdDA6bXNrLWF0RGxuUWllNjhmME9lZTJJcWtwYnRkbDE1bEo=",
            base_url="https://api.moonshot.cn/v1")
    else:
        client = OpenAI(
            api_key="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",
        )

    # with open('data_for_llm_scoring.json', encoding="utf-8") as file_obj:
    with open('test_set_reward3.json', encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    res_tot = []
    idx = 0
    random.seed(42)
    random_list = random.sample(range(len(data)), 100)

    for idx, data_point in enumerate(data):
        en = data_point['en']
        par = data_point['par']
        res = {}
        res['en'] = en
        res['zh'] = []
        res['scores'] = []
        res['llama_1'] = []
        res['pairs'] = []
        res['idx'] = idx
        # if idx not in random_list:
        #     continue
        # for zh in data_point['kimi']:
        zh = data_point['zh']
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
            response = eval_llama([prompt], policy_model, tokenizer, N) if use_llama else eval_gpt(prompt, N)
        except:
            continue
        
        r = 0
        for i in range(N):
            score = [float(n) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', response[i])][0]
            r += score
        r /= N
        
        llama_list_reward3.append(int(round(r)))
        human_score = min(max(data_point['score3'], 2), 3)
        human_list_reward3.append(human_score)
        print("idx=", idx, "en=", en, "zh=", zh, "reward3 score=", r, int(round(r)), 'human score=', human_score)
        
        idx += 1
        print("---------------------------", idx, "finished-----------------------------------")

    # print(res_tot)
    print("begin to get corelation")
    if False:
        get_corelation(human_list[0], llama_list[0], 'human vs llama scores on reward1', 'human scores', 'llama scores', 'human_llama_reward1.png')
        get_corelation(human_list[1], llama_list[1], 'human vs llama scores on reward2', 'human scores', 'llama scores', 'human_llama_reward2.png')
        get_corelation(human_list[2], llama_list[2], 'human vs llama scores on reward3', 'human scores', 'llama scores', 'human_llama_reward3.png')
    else:
        get_corelation(human_list_reward3, llama_list_reward3, 'human vs llama on reward3', 'human scores', 'llama scores', 'human_llama_reward3.png')
        with open('llama_test_scores.json', 'w', encoding='utf-8') as file_obj:
            json.dump(llama_list_reward3, file_obj, ensure_ascii=False)
        with open('human_test_scores.json', 'w', encoding='utf-8') as file_obj:
            json.dump(human_list_reward3, file_obj, ensure_ascii=False)

def sift_dataset(filename):
    points = [0]*12+[2200000+200000*i for i in range(9)]
    part = 14
    l = points[part]
    r = points[part+1]
    print('range',part,l,r)
    model = EvalReward()
    with open(filename, encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    par_list = []
    zh_list = []
    en_list = []
    for i in range(l,r):
        par_list.append('')
        en_list.append(data[i]['en'])
        zh_list.append(data[i]['zh'])
        '''a slow implementation'''
        # x, y = model.eval_reward12('', data[i]['en'], data[i]['zh']), model.eval_reward3('', data[i]['en'], data[i]['zh'])
        # data[i]["reward_score"] = [x, y]
        # if i % 1000 == 0:
        #     print(i, len(data))
        #     with open(filename[:-5] + "_reward.json", 'w', encoding='utf-8') as file_obj_new:
        #         json.dump(data, file_obj_new, ensure_ascii=False)
    reward12 = model.eval_reward12_batch_samenum(par_list, en_list, zh_list)
    reward3 = model.eval_reward3_batch_samenum(par_list, en_list, zh_list)
    data = {'reward12': reward12, 'reward3': reward3, 'idx': list(range(l,r))}
    with open(filename[:-5] + f"_reward_{part}.json", 'w', encoding='utf-8') as file_obj_new:
        json.dump(data, file_obj_new, ensure_ascii=False)

if __name__ == '__main__':
    # draw_ft()
    # draw_reward3()
    sift_dataset('data/parallel_bt_train.json')