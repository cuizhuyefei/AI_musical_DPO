import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from utils import get_local_dir, TemporarilySeededRandom
from pipeline.utils_pipe import get_rhyme_id
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import json, os
import random

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def strip_html_tags(html_string):
    """Strip HTML tags from a string, except for <code> tags (which contain real code in the StackExchange answers)."""
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_string, 'html.parser')

    # Initialize an empty list to store the text
    text = []
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == 'p':
            text.append(''.join(child.string for child in element.children if isinstance(child, NavigableString)))
        elif element.name == 'pre':
            for code in element.find_all('code'):
                text.append("<code>" + code.get_text() + "</code>")
        elif element.name == 'code':
            text.append("<code>" + element.get_text() + "</code>")

    # Join the text together with newlines in between
    text = "\n\n".join(text)

    return text


def get_se(split, silent=False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the StackExchange dataset from Huggingface, and return a dict of prompts and responses. See get_hh for the format.
    
       We strip the HTML tags from the responses (except for <code> tags), and we add necessary newlines.
    """
    print(f'Loading SE dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('HuggingFaceH4/stack-exchange-preferences', cache_dir=cache_dir)['train']
    print('done')

    # shuffle the dataset and select 1% for test
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(int(len(dataset) * 0.01))) if split == 'test' else dataset.select(
        range(int(len(dataset) * 0.01), len(dataset)))

    def strip_html(x):
        x['question'] = strip_html_tags(x['question'])
        for a in x['answers']:
            a['text'] = strip_html_tags(a['text'])
        return x

    dataset = dataset.map(strip_html, num_proc=64)

    data = defaultdict(dict)
    for row in tqdm.tqdm(dataset, desc='Processing SE', disable=silent):
        prompt = '\n\nHuman: ' + row['question'] + '\n\nAssistant:'
        responses = [' ' + a['text'] for a in row['answers']]
        scores = [a['pm_score'] for a in row['answers']]

        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                pairs.append((i, j) if scores[i] > scores[j] else (j, i))

        data[prompt]['responses'] = responses
        data[prompt]['pairs'] = pairs
        data[prompt]['sft_target'] = max(responses, key=lambda x: scores[responses.index(x)])

    return data

def get_shp(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Stanford Human Preferences dataset from Huggingface and convert it to the necessary format. See hh for the format.

       We filter preference pairs to only keep pairs where the score ratio is at least 2.
       For this dataset, the sft_target is the response with the highest score.
    """
    print(f'Loading SHP dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('stanfordnlp/SHP', split=split, cache_dir=cache_dir)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing SHP', disable=silent):
        prompt = '\n\nHuman: ' + row['history'] + '\n\nAssistant:'
        responses = [' ' + row['human_ref_A'], ' ' + row['human_ref_B']]
        scores = [row['score_A'], row['score_B']]
        if prompt in data:
            n_responses = len(data[prompt]['responses'])
        else:
            n_responses = 0
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue

        # according to https://huggingface.co/datasets/stanfordnlp/SHP
        data[prompt]['pairs'].append((n_responses, n_responses + 1) if row['labels'] == 1 else (n_responses + 1, n_responses))
        data[prompt]['responses'].extend(responses)
        data[prompt]['scores'].extend(scores)

    for prompt in data:
        data[prompt]['sft_target'] = max(data[prompt]['responses'], key=lambda x: data[prompt]['scores'][data[prompt]['responses'].index(x)])
        del data[prompt]['scores']

    return data


def get_hh(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.
    
       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': List[str],
               'pairs': List[Tuple[int, int]],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
       
       For this dataset, the sft_target is just the chosen response.
    """
    print(f'Loading HH dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)
    print('done')

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex['chosen'])
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing HH', disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen

    return data

def get_lyricslength(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading lyricslength dataset ({split} split) from offline...')
    with open('./data/lyricslength_{}.json'.format(split), encoding="utf-8") as file_obj:
        dataset = json.load(file_obj)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    fir = True
    for row in dataset:
        s = row['prompt']
        if s[-2] == ' ':
            num = int(s[-1])
            en_sentence = s[:-3]
        else: 
            num = int(s[-2:])
            en_sentence = s[:-4]
        if en_sentence[-1] == '\t':
            en_sentence = en_sentence[-1]
        #len_lst = [1 for i in range(num)]
        prompt = f'''I will give you a English lyric, and you need to translation it into Chinese with exactly {num} characters. Please only output the translated results and nothing more.  \
The English lyrics is: {en_sentence}. Then the translation result is:'''
        if fir:
            print("prompt = ", prompt)
            fir = False
        data[prompt]['pairs'] = [(0, 1)]
        data[prompt]['responses'] = row['responses']
        data[prompt]['sft_target'] = row['sft_target']
    print("get lyricslength ok!")
    return data

def get_parallel_translation(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading parallel_translation dataset ({split} split) from offline...')
    with open('./data/parallel_{}.json'.format(split), encoding="utf-8") as file_obj:
        dataset = json.load(file_obj)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    fir = True
    for row in dataset:
        en = row['en']
        zh = row['zh']
        # word_boundary = row['word_boundary']
        length = row['len']
        prompt = f'''The Chinese translation of the English lyric "{en}" is:'''
        if fir:
            print("prompt = ", prompt)
            fir = False
        data[prompt]['pairs'] = [(0, 1)]
        data[prompt]['responses'] = ['aaa', 'bbb']
        data[prompt]['sft_target'] = zh
    print("get parallel_translation ok!")
    return data

def get_parallel_bt_translation(split: str, silent: bool = False, cache_dir: str = None, use_filtered_datasets: bool = False) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading parallel_bt_translation dataset ({split} split) from offline...')
    with open('./data/parallel_bt_{}{}.json'.format(split, '_filtered' if use_filtered_datasets else ''), encoding="utf-8") as file_obj:
        dataset = json.load(file_obj)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    fir = True
    for row in dataset:
        en = row['en']
        zh = row['zh']
        # word_boundary = row['word_boundary']
        length = row['len']
        prompt = f'''The Chinese translation of the English lyric "{en}" is:'''
        if fir:
            print("prompt = ", prompt)
            fir = False
        data[prompt]['pairs'] = [(0, 1)]
        data[prompt]['responses'] = ['aaa', 'bbb']
        data[prompt]['sft_target'] = zh
    print("get parallel_bt_translation ok!")
    return data

def get_parallel_length(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading parallel_length dataset ({split} split) from offline...')
    with open('./data/parallel_{}.json'.format(split), encoding="utf-8") as file_obj:
        dataset = json.load(file_obj)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    fir = True
    for row in dataset:
        en = row['en']
        zh = row['zh']
        # word_boundary = row['word_boundary']
        length = row['len']
        prompt = f'''I will give you a English lyric, and you need to translation it into Chinese with exactly {length} characters. Please only output the translated results and nothing more. The English lyrics is: {en}. Then the translation result is:'''
        if fir:
            print("prompt = ", prompt)
            fir = False
        data[prompt]['pairs'] = [(0, 1)]
        data[prompt]['responses'] = ['aaa', 'bbb']
        data[prompt]['sft_target'] = zh
    print("get parallel_length ok!")
    return data

def get_parallel_bt_length(split: str, silent: bool = False, cache_dir: str = None, use_filtered_datasets: bool = False) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading parallel_bt_length dataset ({split} split) from offline...')
    with open('./data/parallel_bt_{}{}.json'.format(split, '_filtered' if use_filtered_datasets else ''), encoding="utf-8") as file_obj:
        dataset = json.load(file_obj)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    fir = True
    for row in dataset:
        en = row['en']
        zh = row['zh']
        # word_boundary = row['word_boundary']
        length = row['len']
        prompt = f'''I will give you a English lyric, and you need to translation it into Chinese with exactly {length} characters. Please only output the translated results and nothing more. The English lyrics is: {en}. Then the translation result is:'''
        if fir:
            print("prompt = ", prompt)
            fir = False
        data[prompt]['pairs'] = [(0, 1)]
        data[prompt]['responses'] = ['aaa', 'bbb']
        data[prompt]['sft_target'] = zh
    print("get parallel_bt_length ok!")
    return data

def get_parallel_bt_rhyme(split: str, silent: bool = False, cache_dir: str = None, use_filtered_datasets: bool = False, random_drop_rhyme=0.) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading parallel_bt_rhyme dataset ({split} split) from offline...')
    if random_drop_rhyme > 0:
        print(f'random_drop_rhyme = {random_drop_rhyme}')
    dataset_type = ''
    if use_filtered_datasets==1:
        dataset_type = '_filtered'
    elif use_filtered_datasets==2:
        dataset_type = '_HQ'
    elif use_filtered_datasets==3:
        dataset_type = '_accurate'
    with open('./data/parallel_bt_{}{}.json'.format(split, dataset_type), encoding="utf-8") as file_obj:
        dataset = json.load(file_obj)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    fir = True
    for row in dataset:
        en = row['en']
        zh = row['zh']
        # word_boundary = row['word_boundary']
        length = row['len']
        rhyme = get_rhyme_id(zh)
        if random.random() < random_drop_rhyme:
            prompt = f'''I will give you a English lyric, and you need to translation it into Chinese with exactly {length} characters. Please only output the translated results and nothing more. The English lyrics is: {en}. Then the translation result is:'''
        else:
            prompt = f'''I will give you a English lyric, and you need to translation it into Chinese with exactly {length} characters, where the ending rhyme type is {rhyme}. Please only output the translated results and nothing more. The English lyrics is: {en}. Then the translation result is:'''
        if fir:
            print("prompt = ", prompt)
            fir = False
        data[prompt]['pairs'] = [(0, 1)]
        data[prompt]['responses'] = ['aaa', 'bbb']
        data[prompt]['sft_target'] = zh
    print("get parallel_bt_rhyme ok!")
    return data

def get_quality_length(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading quality_length dataset ({split} split) from offline...')
    with open('./data/quality_{}.json'.format(split), encoding="utf-8") as file_obj:
        dataset = json.load(file_obj)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    fir = True
    for row in dataset:
        en = row['en']
        zh = row['zh']
        if len(zh) == 0 or len(row['pairs']) == 0:
            continue
        length = len(zh[0])
        prompt = f'''I will give you a English lyric, and you need to translation it into Chinese with exactly {length} characters. Please only output the translated results and nothing more. The English lyrics is: {en}. Then the translation result is:'''
        if fir:
            print("prompt = ", prompt)
            fir = False
        data[prompt]['pairs'] = row['pairs']
        data[prompt]['responses'] = zh
        data[prompt]['sft_target'] = zh[-1]
    print("get parallel_length ok!")
    return data

# DPO
def get_llama_quality_length(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading llamaquality_length dataset ({split} split) from offline...')
    with open('./data/llamaquality_{}.json'.format(split), encoding="utf-8") as file_obj:
        dataset = json.load(file_obj)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    fir = True
    for row in dataset:
        en = row['en']
        zh = row['zh']
        if len(zh) == 0 or len(row['pairs']) == 0:
            continue
        length = len(zh[0])
        prompt = f'''I will give you a English lyric, 
and you need to translation it into Chinese with exactly {length} characters. 
Please only output the translated results and nothing more. 
The English lyrics is: {en}. Then the translation result is:'''
        if fir:
            print("prompt = ", prompt)
            fir = False
        data[prompt]['pairs'] = row['pairs']
        data[prompt]['responses'] = zh
        data[prompt]['sft_target'] = zh[-1]
    print("get llamaquality_length ok!")
    return data

def get_test_musical(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading test_musical dataset ({split} split) from offline...')
    # with open('./data/musical_test.json', encoding="utf-8") as file_obj:
    #     dataset = json.load(file_obj)
    
    with open('./data/parallel_test.json', encoding="utf-8") as file_obj:
        dataset = json.load(file_obj)
    # with open('final_test.json', encoding="utf-8") as file_obj:
    #     dataset = json.load(file_obj)
    # with open('test_set_reward3.json', encoding="utf-8") as file_obj:
    #     dataset2 = json.load(file_obj)
    #     dataset.extend(dataset2)
    print('input done', len(dataset))

    data = defaultdict(lambda: defaultdict(list))
    fir = True
    for row in dataset:
        en = row['en']
        zh = row['zh']
        trans = zh
        for ch in ',./，。、！!?？：:':
            trans = trans.replace(ch, '')
        length = len(trans)
        prompt = f'''I will give you a English lyric, and you need to translation it into Chinese with exactly {length} characters. Please only output the translated results and nothing more. The English lyrics is: {en}. Then the translation result is:'''
        if fir:
            print("prompt = ", prompt)
            fir = False
        data[prompt]['pairs'] = [(0, 1)]
        data[prompt]['responses'] = ['aaa', 'bbb']
        data[prompt]['sft_target'] = zh
    print("get test_musical ok with size", len(data))
    return data

def get_test_musical_rhyme(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading test_musical_rhyme dataset ({split} split) from offline...')
    # with open('./data/musical_test.json', encoding="utf-8") as file_obj:
    #     dataset = json.load(file_obj)
    
    with open('./data/parallel_test.json', encoding="utf-8") as file_obj:
        dataset = json.load(file_obj)
    
    # with open('final_test.json', encoding="utf-8") as file_obj:
    #     dataset = json.load(file_obj)
    # with open('test_set_reward3.json', encoding="utf-8") as file_obj:
    #     dataset2 = json.load(file_obj)
    #     dataset.extend(dataset2)
    print('input done', len(dataset))

    data = defaultdict(lambda: defaultdict(list))
    fir = True

    occur_norhyme = {}
    for row in dataset:
        en = row['en']
        zh = row['zh']
        trans = zh
        for ch in ',./，。、！!?？：:':
            trans = trans.replace(ch, '')
        length = len(trans)
        rhyme = get_rhyme_id(zh)
        if en + str(length) in occur_norhyme:
            continue
        occur_norhyme[en + str(length)] = True
        prompt = f'''I will give you a English lyric, and you need to translation it into Chinese with exactly {length} characters, where the ending rhyme type is {rhyme}. Please only output the translated results and nothing more. The English lyrics is: {en}. Then the translation result is:'''
        if fir:
            print("prompt = ", prompt)
            fir = False
        data[prompt]['pairs'] = [(0, 1)]
        data[prompt]['responses'] = ['aaa', 'bbb']
        data[prompt]['sft_target'] = zh
    print("get test_musical_rhyme ok with size", len(data))
    return data

def get_pure_length(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading purelength dataset ({split} split) from offline...')
    with open('./data/purelength_{}.json'.format(split), encoding="utf-8") as file_obj:
        dataset = json.load(file_obj)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    fir = True
    for row in dataset:
        en = row['en']
        zh = row['zh']
        if len(zh) == 0 or len(row['pairs']) == 0:
            continue
        length = len(zh[0])
        prompt = f'''I will give you a English lyric, and you need to translation it into Chinese with exactly {length} characters. Please only output the translated results and nothing more. The English lyrics is: {en}. Then the translation result is:'''
        if fir:
            print("prompt = ", prompt)
            fir = False
        data[prompt]['pairs'] = row['pairs']
        data[prompt]['responses'] = zh
        data[prompt]['sft_target'] = zh[-1]
    print("get purelength ok!")
    return data

def get_parallel_word_boundary(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading parallel_word_boundary dataset ({split} split) from offline...')
    with open('./data/parallel_{}.json'.format(split), encoding="utf-8") as file_obj:
        dataset = json.load(file_obj)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    fir = True
    for row in dataset:
        en = row['en']
        zh = row['zh']
        word_boundary = row['word_boundary']
        length = row['len']
        prompt = f'''I will give you a English lyric, and you need to translation it into Chinese with exactly {length} characters. The word boundary should be {word_boundary}, where 1 means "there should be a word boundary after this syllable" and 0 means "we do not care if there is a boundary". Please only output the translated results and nothing more. The English lyrics is: {en}. Then the translation result is:'''
        if fir:
            print("prompt = ", prompt)
            fir = False
        data[prompt]['pairs'] = [(0, 1)]
        data[prompt]['responses'] = ['aaa', 'bbb']
        data[prompt]['sft_target'] = zh
    print("get parallel_word_boundary ok!")
    return data

def get_parallel_translation_zh(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading parallel_translation dataset ({split} split) from offline...')
    with open('./data/parallel_{}.json'.format(split), encoding="utf-8") as file_obj:
        dataset = json.load(file_obj)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    fir = True
    for row in dataset:
        en = row['en']
        zh = row['zh']
        word_boundary = row['word_boundary']
        length = row['len']
        prompt = f'''英文歌词“{en}”的中文翻译是:'''
        if fir:
            print("prompt = ", prompt)
            fir = False
        data[prompt]['pairs'] = [(0, 1)]
        data[prompt]['responses'] = ['aaa', 'bbb']
        data[prompt]['sft_target'] = zh
    print("get parallel_translation ok!")
    return data

def get_parallel_length_zh(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading parallel_length dataset ({split} split) from offline...')
    with open('./data/parallel_{}.json'.format(split), encoding="utf-8") as file_obj:
        dataset = json.load(file_obj)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    fir = True
    for row in dataset:
        en = row['en']
        zh = row['zh']
        word_boundary = row['word_boundary']
        length = row['len']
        prompt = f'''我会给你一句英文歌词，你需要将其翻译成中文，精确到{length}个字。请仅输出翻译结果。英文歌词是：{en}，则翻译结果是：'''
        if fir:
            print("prompt = ", prompt)
            fir = False
        data[prompt]['pairs'] = [(0, 1)]
        data[prompt]['responses'] = ['aaa', 'bbb']
        data[prompt]['sft_target'] = zh
    print("get parallel_length ok!")
    return data

def get_parallel_word_boundary_zh(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading parallel_word_boundary dataset ({split} split) from offline...')
    with open('./data/parallel_{}.json'.format(split), encoding="utf-8") as file_obj:
        dataset = json.load(file_obj)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    fir = True
    for row in dataset:
        en = row['en']
        zh = row['zh']
        word_boundary = row['word_boundary']
        length = row['len']
        prompt = f'''  - 我会给你一句英文歌词，你需要将其翻译成中文，精确到{length}个字。中文词汇的边界应该是{word_boundary}，其中1表示“这个音节后面应该有一个词汇边界”，0表示“我们不关心是否有边界”。请仅输出翻译结果。英文歌词是：{en}，则翻译结果是：'''
        if fir:
            print("prompt = ", prompt)
            fir = False
        data[prompt]['pairs'] = [(0, 1)]
        data[prompt]['responses'] = ['aaa', 'bbb']
        data[prompt]['sft_target'] = zh
    print("get parallel_word_boundary ok!")
    return data

def get_reward1(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    data = defaultdict(lambda: defaultdict(list))
    print(f'Loading reward1 dataset from offline...')
    for i in range(5):
        if not os.path.exists(f'data/data_scores_{i}.json'):
            continue
        with open(f'data/data_scores_{i}.json', encoding="utf-8") as file_obj: #read
            score = json.load(file_obj)
        with open(f'data/scoring_data_{i}.json', encoding="utf-8") as file_obj: #read
            dataset = json.load(file_obj)

        fir = True
        for idx, row in enumerate(dataset):
            if score[idx][0]<1 or score[idx][0]>4: continue
            en = row['en_line']
            zh = row['zh_line']
            context = row['par']
            prompt = f'''You are a Chinese sentence grader. Given Chinese sentence, you need to give scores in range 1-4 (4 is the highest) based on sentence fluency. 

Here are the metrics:
Score 1: Not fluent at all, can't understand the meaning easily.
Score 2: There are inappropriate or awkward phrases or other big unbearable flaws.
Score 3: Quite fluent. There exists small mistakes, but they are acceptable.
Score 4: Very fluent and no mistakes. Likely come from a native Chinese speaker.

Now, I will provide you with the Chinese sentence. You need to give me only one number and nothing else. 
The Chinese sentence is: {zh}. The score is:'''
            if fir:
                # print("prompt = ", prompt)
                fir = False
            data[prompt]['pairs'] = [(0, 1)]
            data[prompt]['responses'] = ['aaa', 'bbb']
            data[prompt]['sft_target'] = str(score[idx][0])
    print("get reward1 ok!")
    return data

def get_reward2(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    data = defaultdict(lambda: defaultdict(list))
    if split != 'train':
        with open('final_test.json', encoding="utf-8") as file_obj:
            dataset = json.load(file_obj)
        for idx, data_point in enumerate(dataset):
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
            
            data[prompt]['pairs'] = [(0, 1)]
            data[prompt]['responses'] = [idx, en, zh]
            data[prompt]['sft_target'] = str(data_point['human_scores'][1])
            # data[prompt]['misc'] = [idx, en, zh]
        print("get reward2 test ok, size =", len(data))
        return data
    
    print(f'Loading reward2 dataset from offline...')
    for i in range(5):
        if not os.path.exists(f'data/data_scores_{i}.json'):
            continue
        with open(f'data/data_scores_{i}.json', encoding="utf-8") as file_obj: #read
            score = json.load(file_obj)
        with open(f'data/scoring_data_{i}.json', encoding="utf-8") as file_obj: #read
            dataset = json.load(file_obj)

        fir = True
        for idx, row in enumerate(dataset):
            if score[idx][1]<1 or score[idx][1]>4: continue
            if score[idx][1] >3: 
                p = random.uniform(0, 1)
                if p > 0.3:
                    continue
            par = row['par']
            en = row['en_line']
            zh = row['zh_line']
            context = row['par']
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
            if fir:
                # print("prompt = ", prompt)
                fir = False
            data[prompt]['pairs'] = [(0, 1)]
            data[prompt]['responses'] = ['aaa', 'bbb']
            data[prompt]['sft_target'] = str(score[idx][1])
    print("get reward2 ok!")
    print(len(data))
    return data

def get_reward3(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    data = defaultdict(lambda: defaultdict(list))
    print(f'Loading reward3 dataset from offline...')
    for i in range(5):
        if not os.path.exists(f'data/data_scores_{i}.json'):
            continue
        with open(f'data/data_scores_{i}.json', encoding="utf-8") as file_obj: #read
            score = json.load(file_obj)
        with open(f'data/scoring_data_{i}.json', encoding="utf-8") as file_obj: #read
            dataset = json.load(file_obj)

        fir = True
        for idx, row in enumerate(dataset):
            if score[idx][2]<1 or score[idx][2]>4: continue
            score[idx][2] = min(max(score[idx][2], 2), 3)
            if score[idx][2] == 2:
                p = random.uniform(0, 1)
                if p > 0.4:
                    continue
            en = row['en_line']
            zh = row['zh_line']
            context = row['par']
            prompt = f'''You are a translation grader. Given a Chinese translation of lyrics, you need to give scores in range 1-4 (4 is the highest) for whether it looks like good lyrics. 

Criteria for scoring:
Score 1: The translation does not resonate as good lyrics.
Score 2: Acceptable as lyrics, but mundane and unremarkable.
Score 3: Good fit for lyrics with some literary flair and aesthetic language.
Score 4: Outstanding lyrical quality, inventive, expressive, and captivating.

Reserve a score of 4 for truly impressive lyricism and be prudent when giving 4. Regular conversational phrases typically merit a score of 2.

Now, I will provide you with the Chinese translation. You need to give me only one number and nothing else. 
The Chinese translation is: {zh}. The score is:'''
            if fir:
                # print("prompt = ", prompt)
                fir = False
            data[prompt]['pairs'] = [(0, 1)]
            data[prompt]['responses'] = ['aaa', 'bbb']
            data[prompt]['sft_target'] = str(score[idx][2])
    print("get reward3 ok!")
    return data

def get_reward_ft(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    data = defaultdict(lambda: defaultdict(list))
    print(f'Loading reward ft dataset from offline...')
    
    with open('data/reward_ft.json', encoding="utf-8") as file_obj:
        dataset = json.load(file_obj)
    for idx, data_point in enumerate(dataset):
        en = data_point['en']
        zh = data_point['zh']
        par = data_point['context']
        
        if data_point['ft_score'] == 4: 
            p = random.uniform(0, 1)
            if p > 0.4:
                continue
        # elif data_point['ft_score'] == 3: 
        #     p = random.uniform(0, 1)
        #     if p > 0.7:
        #         continue
        
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
        
        data[prompt]['pairs'] = [(0, 1)]
        data[prompt]['responses'] = ['aaa', 'bbb']
        data[prompt]['sft_target'] = str(data_point['ft_score'])
    print("get reward ft ok!")
    return data

def get_prompt_refine(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    data = defaultdict(lambda: defaultdict(list))
    print(f'Loading prompt refine dataset from offline...')
    
    with open('/home/zhuorui/translate/prompts_pairs.json', encoding="utf-8") as file_obj:
        dataset = json.load(file_obj)
    for idx, data_point in enumerate(dataset):
        # if split=='test' and idx % 500 != 0:
        #     continue
        prompt = data_point['bad_prompt']
        data[prompt]['pairs'] = [(0, 1)]
        data[prompt]['responses'] = ['aaa', 'bbb']
        data[prompt]['sft_target'] = data_point['good_prompt']
    print("get prompt refine ok!")
    return data

def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None, use_filtered_datasets: int = 0, random_drop_rhyme: float = 0.) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    if name == 'shp':
        data = get_shp(split, silent=silent, cache_dir=cache_dir)
    elif name == 'hh':
        data = get_hh(split, silent=silent, cache_dir=cache_dir)
    elif name == 'se':
        data = get_se(split, silent=silent, cache_dir=cache_dir)
    elif name == 'lyricslength':
        data = get_lyricslength(split, silent=silent, cache_dir=cache_dir)
    elif name == 'paralleltranslation':
        data = get_parallel_translation(split, silent=silent, cache_dir=cache_dir)
    elif name == 'parallelbttranslation':
        data = get_parallel_bt_translation(split, silent=silent, cache_dir=cache_dir, use_filtered_datasets=use_filtered_datasets)
    elif name == 'parallellength':
        data = get_parallel_length(split, silent=silent, cache_dir=cache_dir)
    elif name == 'parallelbtlength':
        data = get_parallel_bt_length(split, silent=silent, cache_dir=cache_dir, use_filtered_datasets=use_filtered_datasets)
    elif name == 'parallelbtrhyme':
        data = get_parallel_bt_rhyme(split, silent=silent, cache_dir=cache_dir, use_filtered_datasets=use_filtered_datasets, random_drop_rhyme=random_drop_rhyme)
    # elif name == 'parallelwordboundary':
    #     data = get_parallel_word_boundary(split, silent=silent, cache_dir=cache_dir)
    # elif name == 'paralleltranslation_zh':
    #     data = get_parallel_translation_zh(split, silent=silent, cache_dir=cache_dir)
    # elif name == 'parallellength_zh':
    #     data = get_parallel_length_zh(split, silent=silent, cache_dir=cache_dir)
    # elif name == 'parallelwordboundary_zh':
    #     data = get_parallel_word_boundary_zh(split, silent=silent, cache_dir=cache_dir)
    elif name == 'qualitylength':    
        data = get_quality_length(split, silent=silent, cache_dir=cache_dir)
    elif name == 'llamaqualitylength':    
        data = get_llama_quality_length(split, silent=silent, cache_dir=cache_dir)
    elif name == 'purelength':
        data = get_pure_length(split, silent=silent, cache_dir=cache_dir)
    elif name == 'reward1':
        data = get_reward1(split, silent=silent, cache_dir=cache_dir)
    elif name == 'reward2':
        data = get_reward2(split, silent=silent, cache_dir=cache_dir)
    elif name == 'reward3':
        data = get_reward3(split, silent=silent, cache_dir=cache_dir)
    elif name == 'reward_ft':
        data = get_reward_ft(split, silent=silent, cache_dir=cache_dir)
    elif name == 'musical_test':
        data = get_test_musical(split, silent=silent, cache_dir=cache_dir)
    elif name == 'musical_test_rhyme':
        data = get_test_musical_rhyme(split, silent=silent, cache_dir=cache_dir)
    # elif name == 'prompt_refine':
    #     data = get_prompt_refine(split, silent=silent, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    assert set(list(data.values())[0].keys()) == {'responses', 'pairs', 'sft_target'}, \
        f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"

    return data


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
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
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
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

    assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

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


def get_batch_iterator(names: List[str],
                       tokenizer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       sft_mode: bool = False,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed:int = 0,
                       silent: bool = False,
                       cache_dir: Optional[str] = None,
                       dataset_clipping:int = 5000000,
                       use_filtered_datasets = 0,
                       random_drop_rhyme = 0.) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()
    print("into get_batch_iterator, split = ", split)
    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            for idx, (prompt, data) in enumerate(get_dataset(name, split, silent=silent, cache_dir=cache_dir, use_filtered_datasets=use_filtered_datasets, random_drop_rhyme=random_drop_rhyme).items()):
                if idx >= dataset_clipping: break
                flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode))
                if data['sft_target'] == '2' and name == 'reward_ft' or data['sft_target'] == '3' and name == 'reward3':
                    p = random.uniform(0, 1)
                    if p < 0.5:
                        flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode))
                    # flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode))
                # if data['sft_target'] == '3':
                #     p = random.uniform(0, 1)
                #     if p > 0.85:
                #         flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode))
                # if data['sft_target'] == '1':
                #     flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode))
        print("already get flat_data, size = ", len(flat_data))

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            # print('[SHUFFLE] debug', next(permutation_seeds), len(flat_data))
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for prompt, responses, pairs, sft_target, truncation_mode in flat_data:
            if done:
                break
            if sft_mode:
                batch_element = tokenize_batch_element(prompt, sft_target, sft_target, truncation_mode, tokenizer, max_length, max_prompt_length)
                batch_element = {k: v for k, v in batch_element.items() if 'rejected' not in k}
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f'Finished generating {n_examples} examples on {split} split')
                        done = True

                    batch = []
            else:
                for p in pairs:
                    if done:
                        break
                    batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length)
                    batch.append(batch_element)
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                            done = True
                        batch = []
        if done:
            break

        epoch_idx += 1


def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise."""
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != ' ' and str_b[idx] != ' ':
                return False
            else:
                if str_a[idx] == ' ':
                    str_a = str_a[:idx] + str_a[idx + 1:]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1:]

    return True