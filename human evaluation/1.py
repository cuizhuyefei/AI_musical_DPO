import pandas as pd
import pingouin as pg
from sklearn.metrics import cohen_kappa_score
import numpy as np
from sklearn.utils import resample

def bootstrap_ci(data, num_bootstraps=1000, ci=90):
    # 计算bootstrap样本的统计量
    boot_means = [np.mean(resample(data)) for _ in range(num_bootstraps)]
    # 计算置信区间边界
    lower_bound = np.percentile(boot_means, (100-ci)/2)
    upper_bound = np.percentile(boot_means, 100-(100-ci)/2)
    return lower_bound, upper_bound

mean_scores = []

def print_list(a):
	for i in range(len(a)):
		for j in range(len(a[i])):
			print(a[i][j], end = '\t')
		print('')
	print('')

ban_list = [13, 14, 15, 17, 19, 20, 22, 24, 28, 30] # no solutions required
datas = [{
    'Object': [],
    'Rater': [],
    'Rating': []
} for i in range(6)] # 6 metrics

for r_id, file_path in enumerate(['1.xlsx', '2.xlsx', '3.xlsx', '4.xlsx']):
	a = [[[] for j in range(6)] for i in range(5)] # 5*6的二维数组，用于存储数据 (5*4 + 5*2)
	print(file_path)
	# 使用read_excel函数读取数据
	df = pd.read_excel(file_path)
	# 查看读取的数据
	# print(df)
	for t in range(30):
		for i in range(5):
			for j in range(4):
				if not pd.isna(df.iloc[0+t*6+i, 2+j]) and not (t+1 in ban_list and j==3):
					a[i][j].append(int(df.iloc[0+t*6+i, 2+j]))
					datas[j]['Object'].append('Obj'+str(i+1)+'_'+str(t+1))
					datas[j]['Rater'].append('Rater'+str(r_id))
					datas[j]['Rating'].append(int(df.iloc[0+t*6+i, 2+j]))
	for t in range(12):
		for i in range(5):
			for j in range(2):
				if not pd.isna(df.iloc[184+t*6+i, 2+j]):
					a[i][4+j].append(int(df.iloc[184+t*6+i, 2+j]))
					datas[4+j]['Object'].append('Obj'+str(i+1)+'_'+str(30+t+1))
					datas[4+j]['Rater'].append('Rater'+str(r_id))
					datas[4+j]['Rating'].append(int(df.iloc[184+t*6+i, 2+j]))
	len_a = [[len(a[i][j]) for j in range(6)] for i in range(5)]
	print(len_a)
	mean_a = [[round(sum(a[i][j])/len(a[i][j]), 2) for j in range(6)] for i in range(5)]
	mean_scores.append(mean_a)
mean_score = [[round(sum([mean_scores[i][j][k] for i in range(len(mean_scores))])/len(mean_scores), 2) for k in range(6)] for j in range(5)]
ci_score = [[bootstrap_ci([mean_scores[i][j][k] for i in range(len(mean_scores))]) for k in range(6)] for j in range(5)]
ci_score = [[(round(ci_score[i][j][0], 2), round(ci_score[i][j][1], 2)) for j in range(6)] for i in range(5)]
print_list(mean_score)
print_list(ci_score)

for data in datas:
	df = pd.DataFrame(data)

	# 计算 ICC3k
	icc_result = pg.intraclass_corr(data=df, raters='Rater', targets='Object', ratings='Rating').round(3)
	# print(icc_result)
	# 打印 ICC3k 的结果
	icc3k_result = icc_result[icc_result['Type'] == 'ICC3k']
	print(icc3k_result)


# 将二维数组转换为DataFrame
df = pd.DataFrame(mean_score)
# 保存为CSV文件
df.to_csv('output.csv', index=False)  # 设置index=False以避免将行索引也写入CSV文件


# import pandas as pd
# import pingouin as pg

# # 示例数据，长格式
# data = {
#     'Object': ['Obj1', 'Obj1', 'Obj1', 'Obj2', 'Obj2', 'Obj2', 'Obj3', 'Obj3', 'Obj3', 'Obj4', 'Obj4', 'Obj4'],
#     'Rater': ['Rater1', 'Rater2', 'Rater3', 'Rater1', 'Rater2', 'Rater3', 'Rater1', 'Rater2', 'Rater3', 'Rater1', 'Rater2', 'Rater3'],
#     'Rating': [20, 21, 20, 21, 22, 20, 19, 18, 20, 22, 21, 21]
# }

# df = pd.DataFrame(data)

# # 计算 ICC3k
# icc_result = pg.intraclass_corr(data=df, raters='Rater', targets='Object', ratings='Rating').round(3)

# # 打印 ICC3k 的结果
# icc3k_result = icc_result[icc_result['Type'] == 'ICC3k']
# print(icc3k_result)
