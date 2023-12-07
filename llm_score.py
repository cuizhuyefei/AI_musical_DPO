import re, os, json
from openai import OpenAI

use_kimi = True
N = 20

if use_kimi:
    client = OpenAI(
        api_key="Y2w0dXFxMXI2a2plaXVudDFhdDA6bXNrLWF0RGxuUWllNjhmME9lZTJJcWtwYnRkbDE1bEo=",
        base_url="https://api.moonshot.cn/v1")
else:
    client = OpenAI(
        api_key="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",
    )

with open('data_for_llm_scoring.json', encoding="utf-8") as file_obj:
    data = json.load(file_obj)

res_tot = []
idx = 0
for data_point in data:
    en = data_point['en']
    res = {}
    res['en'] = en
    res['zh'] = []
    res['scores'] = []
    res['pairs'] = []
    for zh in data_point['zh']:
        try:
            response1 = client.chat.completions.create(
                model="moonshot-v1-8k" if use_kimi else "gpt-4",
                messages=[
                {"role": "system", "content": "You should strictly follow the user's requirements."},
                {"role": "user", "content": f'''You are a Chinese sentence grader. Given Chinese sentence, you need to give scores in range 1-4 (4 is the highest) based on sentence fluency. 

            Here are the metrics:
            Score 1: Not fluent at all, can't understand the meaning easily.
            Score 2: There are inappropriate or awkward phrases or other big unbearable flaws.
            Score 3: Quite fluent. There exists small mistakes, but they are acceptable.
            Score 4: Very fluent and no mistakes. Likely come from a native Chinese speaker.
            
            Now, I will provide you with the Chinese sentence. You need to give me only one number and nothing else. 
            The Chinese sentence is: {zh}
            '''}
                ],
                n=N,
                temperature=0.5
            )
        except:
            continue
        try:
            response2 = client.chat.completions.create(
                model="moonshot-v1-8k" if use_kimi else "gpt-4",
                messages=[
                {"role": "system", "content": "You should strictly follow the user's requirements."},
                {"role": "user", "content": f'''You are a translation grader. Given English lyrics and a corresponding Chinese translation, you need to give scores in range 1-4 (4 is the highest) for translation accuracy. 

            Here are the metrics for translation accuracy:
            Score 1: More than 50% is translated wrongly, or there are unbearable translation mistakes.
            Score 2: Merely acceptable, but there are mistakes that need correction.
            Score 3: No big mistake in translation, totally acceptable. But there is still space for improvement, such as phrasing or the translation of idioms.
            Score 4: Excellent translation.

            Note that in either metrics, score 4 means excellent and should be only given if you are absolutely sure the translated sentence is perfect. Any tiny mistake will make its score less than 4.

            Now, I will provide you with the English lyrics and the Chinese translation. You need to give me only one number and nothing else. 
            The English lyrics is: {en}
            The Chinese translation is: {zh}
            '''}
                ],
                n=N,
                temperature=0.5
            )
        except:
            continue
        r1 = 0
        r2 = 0
        for i in range(N):
            score1 = [float(n) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', response1.choices[i].message.content)][0]
            score2 = [float(n) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', response2.choices[i].message.content)][0]
            r1 += score1
            r2 += score2
        r1 /= N
        r2 /= N
        print("en=", en, "zh=", zh, "fluency=", r1, "translation=", r2)
        res['zh'].append(zh)
        res['scores'].append((r1, r2))
    num = len(res['zh'])
    for i in range(num):
        for j in range(i):
            d1 = res['scores'][i][0] - res['scores'][j][0]
            d2 = res['scores'][i][1] - res['scores'][j][1]
            if d1 > 0.5 and d2 > 0.5:
                res["pairs"].append((i, j))
            elif d1 < -0.5 and d2 < -0.5:
                res["pairs"].append((j, i))
            elif d1 > 0 and d2 > 1.0:
                res["pairs"].append((i, j))
            elif d1 < 0 and d2 < -1.0:
                res["pairs"].append((j, i))
            elif d1 > 1.0 and d2 > 0:
                res["pairs"].append((i, j))
            elif d1 < -1.0 and d2 < 0:
                res["pairs"].append((j, i))
    res_tot.append(res)
    with open('llm_scores.json', 'w', encoding='utf-8') as file_obj:
        json.dump(res_tot, file_obj, ensure_ascii=False)
    idx += 1
    print("---------------------------", idx, "finished-----------------------------------")

print(res_tot)    


# data = [
#     {
#         'en': "I have never been satisfied.", 
#         'zh': ["我从未满足过自己的", "我从来都不曾满足了", "我从来都不曾满足过", "我从来都不满足过去", "我从来都不会满足了", "我从未满足，始终如此"],
#     },
#     {
#         'en': 'The sun is shining,',
#         'zh': ['太阳在闪', '阳光灿烂'],
#     },
#     {
#         'en': 'When you are sixteen going on seventeen.',
#         'zh': ["当你六十一岁去年满十七", "当你已经六十一岁而未满", "当你年纪是十六岁又一岁", "当你十六岁快要十七岁时", "当你十六岁，迈向十七岁轮", "当你十六岁，迈向十七岁"],
#     },
#     {
#         'en': 'How do you keep a wave upon the sand?',
#         'zh': ["任何地方你去我跟去", "你如何保持沙滩的浪花", "你是如何在沙滩上守望", "你怎么能把海浪留在沙", "你怎么能让浪花停在沙", "你如何在沙滩上留下波", "如何让沙上的浪不消散", "如何让沙上的波浪不散", "如何让海浪停留在沙滩"],
#     }
# ]