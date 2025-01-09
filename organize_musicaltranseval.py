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

def process_reward3() -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    output_dataset = []
    print(f'Loading reward3 dataset from offline...')
    for i in range(5):
        if not os.path.exists(f'data/data_scores_{i}.json'):
            continue
        with open(f'data/data_scores_{i}.json', encoding="utf-8") as file_obj: #read
            score = json.load(file_obj)
        with open(f'data/scoring_data_{i}.json', encoding="utf-8") as file_obj: #read
            dataset = json.load(file_obj)

        fir = True
        print(len(score), len(dataset))
        for idx, row in enumerate(dataset):
            if score[idx][2]<1 or score[idx][2]>4: continue
            data = {
                "en": row['en_line'],
                "zh": row['zh_line'],
                "score": score[idx][2],
                "musical": row["musical"],
            }
            output_dataset.append(data)
    return output_dataset

# dataset = process_reward3()
# print(len(dataset))
# with open('data/reward_adv.json', 'w', encoding='utf-8') as file_obj:
# 	json.dump(dataset, file_obj, ensure_ascii=False)

def process_reward12() -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    output_dataset = []
    with open('data/reward_ft.json', encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    
    for i in range(len(data)):
        data_new = {
            "en": data[i]["en"],
            "zh": data[i]["zh"],
            "score": data[i]["ft_score"],
            "context": data[i]["context"],
            "musical": data[i]["musical"],
        }
        output_dataset.append(data_new)
    return output_dataset

dataset = process_reward12()
print(len(dataset))
with open('data/reward_bas.json', 'w', encoding='utf-8') as file_obj:
	json.dump(dataset, file_obj, ensure_ascii=False)