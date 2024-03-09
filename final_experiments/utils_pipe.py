from pypinyin import lazy_pinyin
import syllapy
import re, os

'''
rhyme group:
1: a, ia, ua
2: o, e, uo
3: ie, ue
4: zhi, chi, shi, ri, zi, ci, si, er, v, i
5: u
6: ai, uai
7: ei, ui
8: ao, iao
9: ou, iu
10: an, ian, uan, van
11: en, in, un, vn, uen
12: ang, iang, uang
13: eng, ing, ueng, ong, iong
'''

def extract_chinese(input_str):
  # 使用正则表达式匹配中文字符
  chinese_pattern = re.compile('[\u4e00-\u9fa5]+')
  result = chinese_pattern.findall(input_str)
  return ''.join(result)

def get_rhyme_id(str):
    str = extract_chinese(str)
    if len(lazy_pinyin(str, errors='ignore'))==0:
      print(str, lazy_pinyin(str, errors='ignore'))
      return -1
    full = lazy_pinyin(str, errors='ignore')[-1]
    if full == 'n':
      full = 'en'
    if full[0] == 'a' or full[0] == 'o' or full[0] == 'e': # a, ao, an, ang, o, en, eng
        end_rhyme = full
    elif len(full) < 2:
        end_rhyme = -1
        return -1 
    elif full[1] == 'h': # zh, ch, sh
        end_rhyme = full[2:]
    elif full[1] == 'r': # er
        end_rhyme = full
    else:
        end_rhyme = full[1:]
    
    group = [['a', 'ia', 'ua'], 
             ['o', 'e', 'uo'], 
             ['ie', 'ue'],
             ['er', 'v', 'i'],
             ['u'],
             ['ai', 'uai'],
             ['ei', 'ui'],
             ['ao', 'iao'],
             ['ou', 'iu'],
             ['an', 'ian', 'uan', 'van'],
             ['en', 'in', 'un', 'vn', 'uen'],
             ['ang', 'iang', 'uang'],
             ['eng', 'ing', 'ueng', 'ong', 'iong']]
    ans = 0
    for i in range(len(group)):
        if end_rhyme in group[i]:
            ans = i + 1
    # print("for ", str, "end_rhyme is", end_rhyme, "group is", ans)
    return ans

def get_rhyme_rule(idx):
    if idx == 0:
        return -1
    group = [['a', 'ia', 'ua'], 
             ['o', 'e', 'uo'], 
             ['ie', 'ue'],
             ['er', 'v', 'i', 'zhi', 'chi', 'shi', 'ri', 'zi', 'ci', 'si'],
             ['u'],
             ['ai', 'uai'],
             ['ei', 'ui'],
             ['ao', 'iao'],
             ['ou', 'iu'],
             ['an', 'ian', 'uan', 'van'],
             ['en', 'in', 'un', 'vn', 'uen'],
             ['ang', 'iang', 'uang'],
             ['eng', 'ing', 'ueng', 'ong', 'iong']]
    egs = ['啊哈夏花', '我饿国', '杰跃月', '儿绿依事词', '古哭', '帅太', '泪辉', '笑傲', '有秋', '寒先缘', '嗯林', '朗强框', '冷影凶蛹']
    idx -= 1
    str = f'''The rhyme of the last generated Chinese character should fall in {group[idx]}. (Examples of Chinese characters whose rhyme falls in this range is {egs[idx]}).'''
    # print("prompt is", str)
    return str

def get_syllable(s): # wrong cases: debauched 2, bravely 2
  return syllapy.count(s)

def count_length(str):
  for ch in "，。！；？,.!;?()-：:”“…——、 ":    
    str = str.replace(ch, ' ')
  str = str.split(' ')
  return [len(s) for s in str if len(s) > 0]

def count_syllables(input_lyric):
    print("begin count syllables")
    syllables = []
    for line in input_lyric.split('\n'):
        if len(line.strip())==0: continue
        parts = line.replace('\r', '').replace('|', ',').split(',')
        if len(parts[-1]) == 0: parts = parts[:-1]
        numbers = []
        print([line], parts)
        for part in parts:
            syllable = 0
            for s in part.strip('.!?').split(' '):
                if len(s)==0: continue
                syllable += get_syllable(s)
            numbers.append(syllable)
        syllables.append(numbers)
    print(syllables)
    return syllables
  
def cal_par_length(par):
  ret = 0
  for _, line in enumerate(par.split('\n')):
    if len(line.strip())==0: continue
    ret += 1
  return ret