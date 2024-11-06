import os
import numpy as np
import yaml

class DataLoader:

    def __init__(self):
        self.load_data()

    def load_data(self):
        data_path = 'data.yaml'
        with open(data_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        print("length of data: ", len(data['data']))
        for d in data['data']:
            assert('en' in d and 'zh' in d and 'par' in d)
            assert(len(d['zh']) == 5) 

        self.data = data['data']

    def get_data_by_idx(self, en_idx, zh_idx):

        assert((en_idx in range(30)) and (zh_idx in range(5)))

        return {
            'en': self.data[en_idx]['en'],
            'zh': self.data[en_idx]['zh'][zh_idx],
            'par': self.data[en_idx]['par']
        }
    
test_data = DataLoader()
print(test_data.get_data_by_idx(0, 0))