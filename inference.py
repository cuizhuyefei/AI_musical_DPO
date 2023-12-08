import torch
torch.backends.cuda.matmul.allow_tf32 = True
import transformers
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from preference_datasets import get_batch_iterator, get_dataset
from utils import (
    pad_to_length,
    get_local_dir,
)
from typing import Optional, Dict, List, Union, Tuple
import re, json

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

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

def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    """Tokenize a single batch element.
    
       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
       
       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch
def get_collate_fn(tokenizer):
    """Returns a collate function for the given tokenizer.
    
       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]).to('cuda') for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]).to('cuda') for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    return collate_fn
def get_batch_iterator(tokenizer,
                       data,
                       test_data = False,
                       max_length: int = 512,
                       max_prompt_length: int = 128,):
    flat_data = []
    truncation_mode = 'keep_start'
    if test_data:
        for prompt, data_point in data.items():
            flat_data.append((prompt, data_point['responses'], data_point['pairs'], data_point['sft_target'], truncation_mode))
    else:
        for datapoint in data:
            flat_data.append((datapoint['prompt'], datapoint['responses'], 
                            datapoint['pairs'], datapoint['sft_target'], truncation_mode))    
    print("already get flat_data, size = ", len(flat_data))
    
    collate_fn = get_collate_fn(tokenizer)

    batch = []
    for prompt, responses, pairs, sft_target, truncation_mode in flat_data:
        for p in pairs:
            batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]], truncation_mode, 
                                                   tokenizer, max_length, max_prompt_length)
            batch.append(batch_element)
    return collate_fn(batch)

def eval_model(tokenizer, policy_model, data):
    #data = get_dataset(name='parallellength', split='test')
    
    batch = get_batch_iterator(tokenizer, data)
    reference_samples = get_batch_samples(policy_model, tokenizer, batch)
    for prompt, sample in zip(batch['prompt'], reference_samples):
        print("prompt:", prompt)
        print("sample: ", sample)
        print("\n")
    print("------------------------------------------------------------------------")

def prepare_musical_length_batches():
    with open('./data/musical_data.json', encoding="utf-8") as file_obj:
        dataset = json.load(file_obj)
    bsz = 8
    idx = 0
    data = []
    total_data = []
    for row in dataset:
        data_point = {}
        prompt = f'''I will give you a English lyric, and you need to translation it into Chinese with exactly {row['len']} characters. \
Please only output the translated results and nothing more. The English lyrics is: {row['en']}. Then the translation result is:'''
        data_point['prompt'] = prompt
        data_point['responses'] = ['', '']
        data_point['pairs'] = [(0, 1)]
        data_point['sft_target'] = ''
        data_point['all_zh_trans'] = row['zh']
        data.append(data_point)
        idx += 1
        
        if idx % bsz == 0:
            total_data.append(data)
            data = []
    if len(data) > 0:
        total_data.append(data)
    print("total number of batches is", len(total_data))
    return total_data

def run_musical_length_generation(tokenizer, policy_model):
    total_data = prepare_musical_length_batches()
    res = []
    batch_idx = 0
    for data in total_data:
        batch_idx += 1
        batch = get_batch_iterator(tokenizer, data, False)
        reference_samples = get_batch_samples(policy_model, tokenizer, batch, n=5)
        print("batch ", batch_idx, "finished")
        for idx, prompt in enumerate(batch['prompt']):
            desired_length = [float(n) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', prompt)][0]
            desired_gen = []
            for i in range(5*idx, 5*idx+5):
                trans = reference_samples[i][len(prompt):].strip()
                for ch in ',./，。、！!?？：:':
                    trans = trans.replace(ch, '')
                real_length = len(trans)
                if real_length==desired_length:
                    desired_gen.append(reference_samples[i][len(prompt):].strip())
            content = {'prompt': prompt, 'desired_gen': list(set(desired_gen)), 'kimi_gen': data[idx]['all_zh_trans']}
            res.append(content)
            with open('data_for_DPO.json', 'w', encoding='utf-8') as file_obj:
                json.dump(res, file_obj, ensure_ascii=False)
    return res

def cal_length_test_acc(tokenizer, policy_model):
    raw_data = get_dataset(name='parallellength', split='test')
    
    tot = 0
    cor = 0
    err = 0
    
    data = []
    idx = 0
    for prompt, data_point in raw_data.items():
        data_point['prompt'] = prompt
        data.append(data_point)
        idx += 1
        if idx % 16 == 0: 
            batch = get_batch_iterator(tokenizer, data, False)
            print("batch is already generated!", idx)
            reference_samples = get_batch_samples(policy_model, tokenizer, batch)
            #print("generate samples finished!")
            for prompt, sample in zip(batch['prompt'], reference_samples):
                # print("prompt:", prompt)
                # print("sample: ", sample)
                # print("\n")
                trans = sample[len(prompt):]
                trans = trans.strip()
                for ch in ',./，。、！!?？：:':
                    trans = trans.replace(ch, '')
                real_length = len(trans)
                if tot == 0:
                    print("trans = ", trans)
                desired_length = [float(n) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', prompt)]
                tot += 1
                cor += int(real_length == desired_length[0])
                err += abs(real_length - desired_length[0])
            data = []
        if idx % 1000 == 0:
            print("cur idx = ", idx)
    
    if len(data) > 0:
        batch = get_batch_iterator(tokenizer, data, False)
        print("batch is already generated!")
        reference_samples = get_batch_samples(policy_model, tokenizer, batch)
        print("generate samples finished!")
        for prompt, sample in zip(batch['prompt'], reference_samples):
            # print("prompt:", prompt)
            # print("sample: ", sample)
            # print("\n")
            trans = sample[len(prompt):]
            trans = trans.strip()
            for ch in ',./，。、！!?？：:':
                trans = trans.replace(ch, '')
            real_length = len(trans)
            if tot == 0:
                print("trans = ", trans)
            desired_length = [float(n) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', prompt)]
            tot += 1
            cor += int(real_length == desired_length[0])
            err += abs(real_length - desired_length[0])
    print("acc = ", cor / tot, cor, tot)
    print("mean err =", err / tot, err, tot)
    print("------------------------------------------------------------------------")

#loading tokenizer
tokenizer_name_or_path = 'hfl/chinese-alpaca-2-13b'
print(f'Loading tokenizer {tokenizer_name_or_path}')
tokenizer = transformers.LlamaTokenizer.from_pretrained(
    tokenizer_name_or_path, 
    # cache_dir=get_local_dir(['.cache', '/scr-ssd', '/scr'])
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
print("load tokenizer ok!")

# prepare data
data = [{
            'prompt': 'The Chinese translation of the English lyric "You are sixteen, going on seventeen" is:',
            'responses': ['', ''],
            'pairs': [(0, 1)],
            'sft_target': '',
        },
        {
            'prompt': 'The Chinese translation of the English lyric "Even when the dark comes crushing through" is:',
            'responses': ['', ''],
            'pairs': [(0, 1)],
            'sft_target': '',    
        },
        {
            'prompt': 'I will give you a English lyric, and you need to translation it into Chinese with exactly 8 characters. \
Please only output the translated results and nothing more. The English lyrics is: You are sixteen, going on seventeen. Then the translation result is:',
            'responses': ['', ''],
            'pairs': [(0, 1)],
            'sft_target': '',
        },
        {
            'prompt': 'I will give you a English lyric, and you need to translation it into Chinese with exactly 9 characters. \
Please only output the translated results and nothing more. The English lyrics is: Even when the dark comes crushing through. Then the translation result is:',
            'responses': ['', ''],
            'pairs': [(0, 1)],
            'sft_target': '',    
        },
        {
            'prompt': 'I will give you a English lyric, and you need to translation it into Chinese with exactly 8 characters. \
The word boundary should be 00010000, where 1 means "there should be a word boundary after this syllable" and 0 means "we do not care if there is a boundary".\
Please only output the translated results and nothing more. The English lyrics is: You are sixteen, going on seventeen. Then the translation result is:',
            'responses': ['', ''],
            'pairs': [(0, 1)],
            'sft_target': '',
        },
        {
            'prompt': 'I will give you a English lyric, and you need to translation it into Chinese with exactly 9 characters. \
The word boundary should be 000000000, where 1 means "there should be a word boundary after this syllable" and 0 means "we do not care if there is a boundary".\
Please only output the translated results and nothing more. The English lyrics is: Even when the dark comes crushing through. Then the translation result is:',
            'responses': ['', ''],
            'pairs': [(0, 1)],
            'sft_target': '',    
        },
        {
            'prompt': '英文歌词“You are sixteen, going on seventeen”的中文翻译是:',
            'responses': ['', ''],
            'pairs': [(0, 1)],
            'sft_target': '',
        },
        {
            'prompt': '英文歌词“Even when the dark comes crushing through”的中文翻译是:',
            'responses': ['', ''],
            'pairs': [(0, 1)],
            'sft_target': '',    
        },
        {
            'prompt': '我会给你一句英文歌词，你需要将其翻译成中文，精确到8个字。\
请仅输出翻译结果。英文歌词是：You are sixteen, going on seventeen，则翻译结果是：',
            'responses': ['', ''],
            'pairs': [(0, 1)],
            'sft_target': '',
        },
        {
            'prompt': '我会给你一句英文歌词，你需要将其翻译成中文，精确到9个字。\
请仅输出翻译结果。英文歌词是：Even when the dark comes crushing through，则翻译结果是：',
            'responses': ['', ''],
            'pairs': [(0, 1)],
            'sft_target': '',    
        },{
            'prompt': '  - 我会给你一句英文歌词，你需要将其翻译成中文，精确到8个字。\
中文词汇的边界应该是00010000，其中1表示“这个音节后面应该有一个词汇边界”，0表示“我们不关心是否有边界”。请仅输出翻译结果。\
英文歌词是：You are sixteen, going on seventeen，则翻译结果是：',
            'responses': ['', ''],
            'pairs': [(0, 1)],
            'sft_target': '',
        },
        {
            'prompt': '  - 我会给你一句英文歌词，你需要将其翻译成中文，精确到9个字。\
中文词汇的边界应该是000000000，其中1表示“这个音节后面应该有一个词汇边界”，0表示“我们不关心是否有边界”。请仅输出翻译结果。\
英文歌词是：Even when the dark comes crushing through，则翻译结果是：',
            'responses': ['', ''],
            'pairs': [(0, 1)],
            'sft_target': '',    
        },
    ]
print("prepare data finished!")

# loading model
policy_model_dtype = getattr(torch, 'float32')
policy_model = transformers.AutoModelForCausalLM.from_pretrained(
    'hfl/chinese-alpaca-2-13b', 
    cache_dir=get_local_dir(['.cache',]), 
    low_cpu_mem_usage=True, 
    torch_dtype=policy_model_dtype)
disable_dropout(policy_model)
device = 'cuda'
policy_model = policy_model.to(device)


length_data = [{
            'prompt': 'I will give you a English lyric, and you need to translation it into Chinese with exactly 8 characters. \
Please only output the translated results and nothing more. The English lyrics is: You are sixteen, going on seventeen. Then the translation result is:',
            'responses': ['', ''],
            'pairs': [(0, 1)],
            'sft_target': '',
        },
        {
            'prompt': 'I will give you a English lyric, and you need to translation it into Chinese with exactly 9 characters. \
Please only output the translated results and nothing more. The English lyrics is: Even when the dark comes crushing through. Then the translation result is:',
            'responses': ['', ''],
            'pairs': [(0, 1)],
            'sft_target': '',    
        },
        {
            'prompt': 'I will give you a English lyric, and you need to translation it into Chinese with exactly 8 characters. \
Please only output the translated results and nothing more. The English lyrics is: Even when the dark comes crushing through. Then the translation result is:',
            'responses': ['', ''],
            'pairs': [(0, 1)],
            'sft_target': '',    
        },
        {
            'prompt': 'I will give you a English lyric, and you need to translation it into Chinese with exactly 9 characters. \
Please only output the translated results and nothing more. The English lyrics is: You are sixteen, going on seventeen. Then the translation result is:',
            'responses': ['', ''],
            'pairs': [(0, 1)],
            'sft_target': '',    
        },
        {
            'prompt': 'I will give you a English lyric, and you need to translation it into Chinese with exactly 14 characters. \
Please only output the translated results and nothing more. The English lyrics is: My soul is spiraling in frozen fractals all around. Then the translation result is:',
            'responses': ['', ''],
            'pairs': [(0, 1)],
            'sft_target': '',    
        },
        {
            'prompt': 'I will give you a English lyric, and you need to translation it into Chinese with exactly 9 characters. \
Please only output the translated results and nothing more. The English lyrics is: This is my Quest to follow that star. Then the translation result is:',
            'responses': ['', ''],
            'pairs': [(0, 1)],
            'sft_target': '',    
        },
]

# eval_model(tokenizer, policy_model, length_data)
# cal_length_test_acc(tokenizer, policy_model)
state_dict = torch.load('.cache/zhuorui/parallel_length_sft_2023-12-04_05-49-24_265219/LATEST/policy.pt', map_location='cpu')
policy_model.load_state_dict(state_dict['state'])
policy_model = policy_model.to(device)
# cal_length_test_acc(tokenizer, policy_model)
# eval_model(tokenizer, policy_model, length_data)

res = run_musical_length_generation(tokenizer, policy_model)
with open('data_for_DPO.json', 'w', encoding='utf-8') as file_obj:
    json.dump(res, file_obj, ensure_ascii=False)

# print("original model:---------------------------------------")
# eval_model(tokenizer, policy_model, data)

# print("translation:---------------------------------------")
# state_dict = torch.load('.cache/zhuorui/parallel_translation_sft_2023-12-03_22-43-13_740607/LATEST/policy.pt', map_location='cpu')
# policy_model.load_state_dict(state_dict['state'])
# eval_model(tokenizer, policy_model, data)

# print("translation_zh:---------------------------------------")
# state_dict = torch.load('.cache/zhuorui/parallel_translation_zh_sft_2023-12-04_01-42-01_195500/LATEST/policy.pt', map_location='cpu')
# policy_model.load_state_dict(state_dict['state'])
# eval_model(tokenizer, policy_model, data)

# print("length:---------------------------------------")
# state_dict = torch.load('.cache/zhuorui/parallel_length_sft_2023-12-04_05-49-24_265219/LATEST/policy.pt', map_location='cpu')
# policy_model.load_state_dict(state_dict['state'])
# eval_model(tokenizer, policy_model, data)

# print("length_zh:---------------------------------------")
# state_dict = torch.load('.cache/zhuorui/parallel_length_zh_sft_2023-12-04_08-00-58_939504/LATEST/policy.pt', map_location='cpu')
# policy_model.load_state_dict(state_dict['state'])
# eval_model(tokenizer, policy_model, data)

# print("word_boundary:---------------------------------------")
# state_dict = torch.load('.cache/zhuorui/parallel_word_boundary_sft_2023-12-04_06-25-55_149065/LATEST/policy.pt', map_location='cpu')
# policy_model.load_state_dict(state_dict['state'])
# eval_model(tokenizer, policy_model, data)

# print("word_boundary_zh:---------------------------------------")
# state_dict = torch.load('.cache/zhuorui/parallel_word_boundary_zh_sft_2023-12-04_07-11-51_477085/LATEST/policy.pt', map_location='cpu')
# policy_model.load_state_dict(state_dict['state'])
# eval_model(tokenizer, policy_model, data)