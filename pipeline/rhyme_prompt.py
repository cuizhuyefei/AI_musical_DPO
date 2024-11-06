from pypinyin import lazy_pinyin
import syllapy
import re, os, sys, json
import time
from lyrics_info import Man, LetItGo, ForTheFirstTimeInForever, YouAreNotAlone, SixteenGoingOn17, TheImpossibleDream, MeAndTheSky, ExWives, Phantom, Matilda
from lyrics_info import AvenueQ, YouAreNotAlone, OnlyUs, IMissTheMountains, YouDontKnow, Popular, RevoltingChildren, Six, WavingThroughAWindow, Hurricane, WaitForIt
from utils_pipe import extract_chinese, get_rhyme_id, get_rhyme_rule, get_syllable, count_length, count_syllables, cal_par_length
from openai import OpenAI
import torch

def eval_gpt(prompt, N, T=0.7):
    assert use_kimi
    for kase in range(5):
      try:
        response = client.chat.completions.create(
            model="moonshot-v1-8k" if not use_gpt4o_instead else "gpt-4o",
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

pure_gpt = True
few_shot_prompt = True
def generate_par(par, syllables, use_kimi, use_reward, N, N2, verbose=True):
  if pure_gpt:
    par_list = [s for s in par.split('\n') if len(s.strip())>0]
    par = '\n'.join(par_list)
    assert len(par_list) == len(syllables)
    numbers_list = [sum(syllables[i]) for i in range(len(par_list))]
    while True:
      prompt_kimi = f'''Please translate the following English paragraph into Chinese, adhering strictly to the specified number of Chinese characters per line (commas do not count towards the character count). Maintain a strict line-by-line correspondence between the English and Chinese versions, ensuring the number of lines remains the same.

  The English paragraph is: {par}
  The required character count for each line is: {numbers_list}
  IMPORTANT: Output ONLY the translated Chinese paragraph. Do not include any explanations, notes, or additional text. Your translation must strictly follow the given format and character counts.
  The translated version is:'''
      if few_shot_prompt == True:
        prompt_kimi = f'''Please translate the following English paragraph into Chinese, adhering strictly to the specified number of Chinese characters per line (commas do not count towards the character count). Maintain a strict line-by-line correspondence between the English and Chinese versions, ensuring the number of lines remains the same. Some examples:
Example 1:
The English paragraph is: You are sixteen going on seventeen, 
baby it's time to think
Better beware, be canny and careful, 
baby you're on the brink
The required character count for each line is: [10, 8, 9, 9]
The translated version is:
你十六岁，即将要十七岁
宝贝呀，该去思考了
应当警觉、谨慎和当心，
宝贝，你就在危险边缘

Example 2:
The English paragraph is: I am I, Don Quixote, the Lord of La Mancha
My destiny calls and I go
And the wild winds of fortune will carry me onward
Oh, whithersoever they blow
Whithersoever they blow, onward to glory I go
The required character count for each line is: [13, 7, 10, 6, 11]
The translated version is:
正是我，堂吉诃德，拉曼查的英豪
我的宿命召我前进
幸运的狂飙会策我向前
任他风吹雨打
任他风吹雨打，向荣誉进发！

Example 3:
The English paragraph is: Even when the dark comes crashing through
When you need a friend to carry you
And when you're broken on the ground
You will be found
The required character count for each line is: [12, 10, 7, 5]
The translated version is:
即使当你的世界被黑暗吞没
当你需要朋友携手同行
当你摔落在地面
你会被发现

Example 4:
The English paragraph is: Just because you find that life's not fair,
it doesn't mean that you just have to grin and bear it!
If you always take it on the chin and wear it
Nothing will change.
The required character count for each line is: [11, 15, 9, 7]
The translated version is:
若你只是觉得生活不公平，
那并不意味着你必须要微笑着忍受
如果你总是忍气吞声
没有事情会改变

Example 5:
The English paragraph is: and do I dream again?
for now I find
the phantom of the opera is there
inside my mind
The required character count for each line is: [7, 5, 8, 4]
The translated version is:
我是否又做梦了
因为我发现
歌剧魅影就在那里
在我心中

    The English paragraph is: {par}
    The required character count for each line is: {numbers_list}
    IMPORTANT: Output ONLY the translated Chinese paragraph. Do not include any explanations, notes, or additional text. Your translation must strictly follow the given format and character counts.
    The translated version is:'''        
      response = eval_gpt(prompt_kimi, 1, T=0.7)
      response_list = [s for s in response[0].split('\n') if len(s.strip())>0]
      if len(response_list) != len(par_list):
        print('[GG retry]', response_list, par_list, response)
      else:
        break
    assert len(response_list) == len(par_list)
    reward12 = [[0] for i in range(len(par_list))]
    reward3 = [[0] for i in range(len(par_list))]
    # print('DEBUG', N, inputs, '-----------', sols)
    if use_reward:
      reward12 = reward.eval_reward12_batch(par, par_list, [[res] for res in response_list])
      reward3 = reward.eval_reward3_batch(par, par_list, [[res] for res in response_list])
    gen_res = []
    for i in range(len(par_list)):
      data_point = {}
      data_point['en_line'] = par_list[i]
      data_point['zh_trans'] = [response_list[i]]
      data_point['all_zh_trans'] = [response_list[i]]
      data_point['par'] = par
      data_point['len_constraint'] = numbers_list[i]
      data_point['best_rhyme'] = 0
      data_point['rhyme_type'] = [get_rhyme_id(response_list[i])]
      data_point['reward12'] = reward12[i]
      data_point['reward3'] = reward3[i]
      gen_res.append(data_point)
    return '\n'.join(response_list), gen_res

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
  
  # generate more results and choose the best one for each sentence
  result = ''
  gen_res = []
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
    if use_N2:
      prompt_llama = f'''I will give you a English lyric, and you need to translation it into Chinese with exactly {syllables[i]} characters, where the ending rhyme type is {best_rhyme}. Please only output the translated results and nothing more. The English lyrics is: {inputs[i]}. Then the translation result is:'''
      response = llama.call(prompt_llama, N2) if not use_kimi else eval_gpt(prompt, N2, T=0.7)
      sols[i].extend(response)
      if use_reward:
        reward12[i].extend(reward.eval_reward12_batch(par, [inputs[i]], [response])[0])
        reward3[i].extend(reward.eval_reward3_batch(par, [inputs[i]], [response])[0])
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
    result += inputs[i] + '\n' + sols[i][scores.index(min(scores))] + '\t' + f'(rhyme type: {get_rhyme_id(sols[i][scores.index(min(scores))])} scores: {min(scores)} len_constraint: {syllables[i][0]}' + (f' reward12: {reward.eval_reward12(par, inputs[i], sols[i][scores.index(min(scores))])} reward3: {reward.eval_reward3(par, inputs[i], sols[i][scores.index(min(scores))])}' if use_reward else '') + ')\n'
    print('')
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
# AvenueQ(), YouAreNotAlone(), OnlyUs(), IMissTheMountains(), YouDontKnow(), Popular(), RevoltingChildren(), Six()
input_lyric, syllables = AvenueQ()
syllables = [[sum(x)] for x in syllables] # no list, just a number
use_kimi = True # 记得开成False！！
use_gpt4o_instead = True
use_reward = True
use_N2 = False # 40+40
N = 1 # !
N2 = 0
print('N=', N, 'use_N2=', use_N2)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

if use_kimi:
  if use_gpt4o_instead: # use gpt4o
    client = OpenAI()
  else:
    client = OpenAI(
      api_key="Y2w0dXFxMXI2a2plaXVudDFhdDA6bXNrLWF0RGxuUWllNjhmME9lZTJJcWtwYnRkbDE1bEo=",
      base_url="https://api.moonshot.cn/v1")
else:
  from llm_score import llamaGenAPI
  llama = llamaGenAPI() # nofinetune=True for N=20

if use_reward:
  from llm_score import EvalReward
  reward = EvalReward()
#======================================================

if __name__ == '__main__':
  songs = [AvenueQ, YouAreNotAlone, OnlyUs, IMissTheMountains, YouDontKnow, Popular, RevoltingChildren, Six, WavingThroughAWindow, Man, LetItGo, ForTheFirstTimeInForever, SixteenGoingOn17, TheImpossibleDream, MeAndTheSky, ExWives, Phantom, Matilda, Hurricane, WaitForIt]
  # songs = [AvenueQ, YouAreNotAlone, OnlyUs, IMissTheMountains, YouDontKnow]
  songs_results = {}
  all_gen_results = {}
  if os.path.exists('gpt_prompt_5shot_res.json'):
    with open('gpt_prompt_5shot_res.json', encoding="utf-8") as file_obj:
        songs_results = json.load(file_obj)
  if os.path.exists('gpt_prompt_5shot_allres.json'):
    with open('gpt_prompt_5shot_allres.json', encoding="utf-8") as file_obj:
        all_gen_results = json.load(file_obj)
  for songname in songs:
    if songname.__name__ in songs_results:
      print(songname.__name__, "already done")
      continue
    input_lyric, syllables = songname()
    syllables = [[sum(x)] for x in syllables] # no list, just a number
    result, gen_res = generate_song(input_lyric, syllables, use_kimi, use_reward, N, N2)
    songs_results[songname.__name__] = result
    all_gen_results[songname.__name__] = gen_res
    with open('gpt_prompt_5shot_res.json', 'w', encoding='utf-8') as file_obj:
      json.dump(songs_results, file_obj, ensure_ascii=False)
    with open('gpt_prompt_5shot_allres.json', 'w', encoding='utf-8') as file_obj:
      json.dump(all_gen_results, file_obj, ensure_ascii=False)