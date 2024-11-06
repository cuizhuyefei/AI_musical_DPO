import pandas as pd
from sklearn.metrics import cohen_kappa_score
import scipy.stats
import numpy as np

def print_list(a):
	for i in range(len(a)):
		for j in range(len(a[i])):
			print(a[i][j], end = ' ')
		print('')
	print('')

class HumanEvalLoader:
    
    def __init__(self):
        self.data = np.zeros((30, 5, 4, 3)) # number of en, number of zh for this en, number of human scores, dimension of human scores
        self.load_data()

    def load_data(self):

        for file_idx, file_path in enumerate(['1.xlsx', '2.xlsx', '3.xlsx', '4.xlsx']):
            print(file_path)
            
            # 使用read_excel函数读取数据
            df = pd.read_excel(file_path)
            
            for t in range(30): # section 1
                for i in range(5):
                    for j in range(3): # 成句性，准确性，歌词性
                        if not pd.isna(df.iloc[0+t*6+i, 2+j]):
                            self.data[t][i][file_idx][j] = int(df.iloc[0+t*6+i, 2+j])

    def obtain_mean_R_bas(self, en_idx, zh_idx):
        s = 0
        for i in range(4):
            s += np.mean(self.data[en_idx][zh_idx][i][:2])
        return s / 4 
    
    def obtain_mean_R_adv(self, en_idx, zh_idx):
        s = 0
        for i in range(4):
            s += self.data[en_idx][zh_idx][i][2]
        return s / 4
    
eval = HumanEvalLoader()
print(eval.obtain_mean_R_bas(0, 0), eval.obtain_mean_R_adv(1, 1))