from pypinyin import lazy_pinyin
import syllapy
import re, os, sys, json
import time
from utils_pipe import extract_chinese, get_rhyme_id, get_rhyme_rule, get_syllable, count_length, count_syllables, cal_par_length
from openai import OpenAI
import torch

def eval_gpt(prompt, N, T=0.2):
    for kase in range(5):
      try:
        response = client.chat.completions.create(
            model="moonshot-v1-8k" if use_kimi else "gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            n=N,
            temperature=T
        )
        return [response.choices[i].message.content for i in range(N)]
      except:
        time.sleep(5)

def align_score(ref, str):
  def penalty(x):
    return x if x>0 else -2*x
  s = count_length(str)
  if len(ref) > len(s):
    return 100000
  f = [[100000 for i in range(len(s))] for j in range(len(ref))]
  for i in range(len(ref)):
    for j in range(len(s)):
      if i == 0:
        f[i][j] = penalty(ref[i] - sum(s[:j+1]))
      elif j < i:
        f[i][j] = 100000
      else:
        f[i][j] = min(f[i-1][k] + penalty(ref[i] - sum(s[k+1:j+1])) for k in range(j))
  score = f[len(ref)-1][len(s)-1]
  return score

def generate_par(par, syllables, use_kimi, use_reward, N, N2, verbose=False):
  torch.cuda.empty_cache()
  if len(syllables) == 0:
    syllables = count_syllables(par)
  sols = []
  inputs = []
  cnt = 0
  # generate based on word boundary constraints
  for _, line in enumerate(par.split('\n')):
    if len(line.strip())==0: continue
    qwq = '//'
    numbers = sum(syllables[cnt])
    processed_line = line.replace('|', '')
    print("processed_line = ", processed_line)
    prompt_kimi = f'''Please translate an English sentence into Chinese, with the number of Chinese characters as required (commas does not count).
    
Here are some tips to help you meet the strict requirements:
1. you can add words without real meaning, such as "看" "那" "啊" "呦"...
2. you don't have to translate literally and can do paraphrasing: use different expressions to convey the same meaning

Example 1:
The input sentence is "you are sixteen, going on seventeen", the translation requires 10 Chinese characters
The translated version is: 你十六岁，即将要十七岁

Example 2:
The input sentence is "I am I, Don Quixote, the Lord of La Mancha", the translation requires 13 Chinese characters
The translated version is: 正是我，堂吉诃德，拉曼查的英豪

Example 3:
The input sentence is "even when the dark comes crushing through", the translation requires 9 Chinese characters
The translated version is: 就算那黑暗突然袭来

To help you better generate, I will provide the whole context: {par}

You should only output a translation and nothing else. Now, the input sentence is "{processed_line.strip()}", the translation requires {numbers} Chinese characters. The translated version is: 
    '''
    prompt_llama = f'''I will give you a English lyric, and you need to translation it into Chinese with exactly {numbers} characters. Please only output the translated results and nothing more. The English lyrics is: {processed_line.strip()}. Then the translation result is:'''
    response = llama.call(prompt_llama, N) if not use_kimi else eval_gpt(prompt_kimi, N, T=0.7)
    if N==1: response = [response] # ... llama.call() specially check whether N=1
    if verbose: print(line, qwq, numbers, response)
    sols.append(response)
    inputs.append(processed_line.strip())
    cnt += 1
  
  # find the best rhyme using heuristic
  # 0 -> cnt - 1
  best_score = 10**12
  best_rhyme = -1
  print("about to find best rhyme")
  print("first cal reward model")
  reward12 = [[0 for i in range(N)] for i in range(cnt)]
  reward3 = [[0 for i in range(N)] for i in range(cnt)]
  # print('DEBUG', N, inputs, '-----------', sols)
  if use_reward:
    reward12 = reward.eval_reward12_batch(par, inputs, sols)
    reward3 = reward.eval_reward3_batch(par, inputs, sols)
    print("finish calculating rewards")
  for rhyme in range(13):
    score = 0
    for i in range(cnt):
      score += min([3*align_score(syllables[i], sols[i][j])
                    -(2 if get_rhyme_id(sols[i][j])==rhyme else 0)
                    +(10**10 if i==cnt-1 and get_rhyme_id(sols[i][j])!=rhyme else 0)
                    -(reward12[i][j] if use_reward else 0)
                    +(10**10 if use_reward and reward12[i][j]<=2 else 0)
                    -(reward3[i][j] if use_reward else 0) for j in range(N)])
    if score < best_score:
      best_score = score
      best_rhyme = rhyme
  print('best rhyme is', best_rhyme)
  
  # generate more results
  if use_N2:
    sols_new = []
    inputs_new = []
    for i in range(cnt):
      prompt = f'''Please translate an English sentence into Chinese, with the number of Chinese characters as required (commas does not count).

Here are some tips to help you meet the strict requirements:
1. you can add words without real meaning, such as "看" "那" "啊" "呦"...
2. you don't have to translate literally and can do paraphrasing: use different expressions to convey the same meaning

Example 1:
The input sentence is "you are sixteen, going on seventeen", the translation requires 10 Chinese characters. {get_rhyme_rule(get_rhyme_id('岁'))}
The translated version is: 你十六岁，即将要十七岁

Example 2:
The input sentence is "I am I, Don Quixote, the Lord of La Mancha", the translation requires 13 Chinese characters. {get_rhyme_rule(get_rhyme_id('豪'))}
The translated version is: 正是我，堂吉诃德，拉曼查的英豪

Example 3:
The input sentence is "even when the dark comes crushing through", the translation requires 9 Chinese characters. {get_rhyme_rule(get_rhyme_id('来'))}
The translated version is: 就算那黑暗突然袭来

To help you better generate, I will provide the whole context: {par}

You should only output a translation and nothing else. Now, the input sentence is "{inputs[i]}", the translation requires {syllables[i]} Chinese characters. {get_rhyme_rule(best_rhyme)} The translated version is: 
  '''
      prompt_llama = f'''I will give you a English lyric, and you need to translation it into Chinese with exactly {syllables[i]} characters, where the ending rhyme type is {best_rhyme}. Please only output the translated results and nothing more. The English lyrics is: {inputs[i]}. Then the translation result is:'''
      response = llama.call(prompt_llama, N2) if not use_kimi else eval_gpt(prompt, N2, T=0.7)
      # sols[i].extend(response)
      sols_new.append(response)
      inputs_new.append(inputs[i])
    if use_reward:
      # reward12[i].extend(reward.eval_reward12_batch(par, [inputs[i]], [response])[0])
      # reward3[i].extend(reward.eval_reward3_batch(par, [inputs[i]], [response])[0])
      reward12_new = reward.eval_reward12_batch(par, inputs_new, sols_new)
      reward3_new = reward.eval_reward3_batch(par, inputs_new, sols_new)
      for i in range(cnt):
        reward12[i].extend(reward12_new[i])
        reward3[i].extend(reward3_new[i])
        sols[i].extend(sols_new[i])

  # finally generate the result
  result = ''
  gen_res = []
  for i in range(cnt):
    scores = [3*align_score(syllables[i], sols[i][j])
              -(2 if get_rhyme_id(sols[i][j])==best_rhyme else 0)
              +(10**10 if i==cnt-1 and get_rhyme_id(sols[i][j])!=best_rhyme else 0)
              -(reward12[i][j] if use_reward else 0)
              +(10**10 if use_reward and reward12[i][j]<=2 else 0)
              -(reward3[i][j] if use_reward else 0) for j in range(len(sols[i]))]
    if verbose: print("scores = ", scores)
    data_point = {}
    data_point['en_line'] = inputs[i]
    min_indices = [i for i, score in enumerate(scores) if score == min(scores)]
    data_point['zh_trans'] = [sols[i][j] for j in min_indices]
    data_point['all_zh_trans'] = sols[i]
    data_point['par'] = par
    data_point['scores_of_all_trans'] = scores
    data_point['len_constraint'] = syllables[i][0]
    data_point['best_rhyme'] = best_rhyme
    data_point['rhyme_type'] = [get_rhyme_id(sols[i][j]) for j in range(len(sols[i]))]
    data_point['reward12'] = reward12[i]
    data_point['reward3'] = reward3[i]
    gen_res.append(data_point)
    result += inputs[i] + '\n' + sols[i][scores.index(min(scores))] + '\t' + f'(rhyme type: {get_rhyme_id(sols[i][scores.index(min(scores))])} len_constraint: {syllables[i][0]}' + (f' reward12: {reward12[i][scores.index(min(scores))]} reward3: {reward3[i][scores.index(min(scores))]}' if use_reward else '') + f' scores: {min(scores)})\n'
    # print('')
  # print("gen_res of par", par, "= ", gen_res)
  return result, gen_res

def generate_song(input_lyric, syllables, use_kimi, use_reward, N, N2):
  if len(syllables) == 0:
    syllables = count_syllables(input_lyric)
  syllables = [[sum(x)] for x in syllables] # no list, just a number
  
  # split paragraphs
  pars = re.split(r'\n\s*\n', input_lyric)
  print("pars = ", pars)
  
  # generate results for each paragraph, then combine
  idx = 0
  lst = 0
  result = ''
  gen_res = []
  for par in pars:
    par_len = cal_par_length(par)
    idx = lst + par_len
    res, par_res = generate_par(par, syllables[lst: idx], use_kimi, use_reward, N, N2)
    result += res + '\n'
    gen_res.append(par_res)
    lst = idx
    print("par results is", res)
    #break
  
  print(result)
  print(gen_res)
  return result, gen_res

#=========Things need to specify before run============

use_kimi = False # 记得开成False！！
ckpt = 3 # 默认开成3，除非用原模型！！
use_reward = True
use_N2 = True # 80+80
N = 40 # !
N2 = 40
if not use_N2: N2 = 0

tmp = 'kimi' if use_kimi else 'llama'
noreward_suffix = '_noreward' if not use_reward else ''
T_suffix = '_notopp'
filename = f'final_experiments/{tmp}_songs_res_N{N}+{N2}_ckpt{ckpt}{noreward_suffix}{T_suffix}.json'
print(f'N={N}', f'N2={N2}', f'use_kimi={use_kimi}', f'use_reward={use_reward}', f'use_N2={use_N2}', f'filename={filename}')
if ckpt < 3: print(f'[WARN!!] ckpt {ckpt}')

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

if use_kimi:
  client = OpenAI(
    api_key="Y2w0dXFxMXI2a2plaXVudDFhdDA6bXNrLWF0RGxuUWllNjhmME9lZTJJcWtwYnRkbDE1bEo=",
    base_url="https://api.moonshot.cn/v1")
else:
  from llm_score import llamaGenAPI
  llama = llamaGenAPI(ckpt=ckpt) # nofinetune=True for N=20

if use_reward:
  from llm_score import EvalReward
  reward = EvalReward()
#======================================================

if __name__ == '__main__':
  with open('final_experiments/song_lyrics.json', encoding="utf-8") as file_obj:
    info_songs = json.load(file_obj)
  # songs = [AvenueQ, YouAreNotAlone, OnlyUs, IMissTheMountains, YouDontKnow]
  songs_results = {}
  all_gen_results = {}
  if os.path.exists(filename):
    with open(filename, encoding="utf-8") as file_obj:
        songs_results = json.load(file_obj)
  if os.path.exists(filename.replace('res', 'allres')):
    with open(filename.replace('res', 'allres'), encoding="utf-8") as file_obj:
        all_gen_results = json.load(file_obj)
  for song in info_songs:
    if song['name'] in songs_results:
      print(song['name'], "already done")
      continue
    input_lyric = song['lyrics']
    syllables = []
    result, gen_res = generate_song(input_lyric, syllables, use_kimi, use_reward, N, N2)
    songs_results[song['name']] = result
    all_gen_results[song['name']] = gen_res
    with open(filename, 'w', encoding='utf-8') as file_obj:
      json.dump(songs_results, file_obj, ensure_ascii=False)
    with open(filename.replace('res', 'allres'), 'w', encoding='utf-8') as file_obj:
      json.dump(all_gen_results, file_obj, ensure_ascii=False)