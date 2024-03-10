import torch
torch.backends.cuda.matmul.allow_tf32 = True
import transformers
import re, os, json, random
from utils import (
    pad_to_length,
    get_local_dir,
    gpu_memory_usage,
)
from typing import Optional, Dict, List, Union, Tuple
from inference import get_batch_iterator
from cal_corelation import get_corelation
import numpy as np
# from huggingface_hub import push_to_hub
tokenizer_name_or_path = 'hfl/chinese-alpaca-2-13b'
print(f'Loading tokenizer {tokenizer_name_or_path}')
tokenizer = transformers.LlamaTokenizer.from_pretrained(
	tokenizer_name_or_path, 
	# cache_dir=get_local_dir(['.cache', '/scr-ssd', '/scr'])
)
# tokenizer.push_to_hub(repo_id='cuizhuyefei/test')
# exit(0)

# loading model
policy_model_dtype = getattr(torch, 'float32')
print(f'Loading model')
policy_model = transformers.AutoModelForCausalLM.from_pretrained(
		'hfl/chinese-alpaca-2-13b', 
		cache_dir=get_local_dir(['.cache',]), 
		low_cpu_mem_usage=True, 
		torch_dtype=policy_model_dtype)

# for ckpt in range(1,4):
# 	if ckpt==3:
# 		state_dict = torch.load('.cache/zhuorui/C_filtered_sft_2024-02-07_19-29-40_317953/LATEST/policy.pt', map_location='cpu')
# 	elif ckpt==2:
# 		state_dict = torch.load('.cache/zhuorui/C_filtered_sft_2024-02-06_08-15-59_499366/LATEST/policy.pt', map_location='cpu')
# 	elif ckpt==1:
# 		state_dict = torch.load('.cache/zhuorui/C_sft_2024-02-08_07-51-21_911116/LATEST/policy.pt', map_location='cpu')
# 	policy_model.load_state_dict(state_dict['state'])
# 	policy_model.push_to_hub("ckpt"+str(ckpt))

for ckpt in range(2,4):
	if ckpt==3:
		state_dict = torch.load('.cache/zhuorui/reward3_2_2024-01-07_05-23-50_091251/LATEST/policy.pt', map_location='cpu')
	elif ckpt==2:
		state_dict = torch.load('.cache/zhuorui/reward_ft_6_2024-01-03_22-36-12_077105/LATEST/policy.pt', map_location='cpu')
	policy_model.load_state_dict(state_dict['state'])
	policy_model.push_to_hub("reward"+('basic' if ckpt==2 else 'advanced'))

# state_dict = torch.load('.cache/zhuorui/reward3_2_2024-01-07_05-23-50_091251/LATEST/policy.pt', map_location='cpu')
# policy_model.load_state_dict(state_dict['state'])
# policy_model.push_to_hub("reward3")