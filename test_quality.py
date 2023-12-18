import re, os, json
import time
from openai import OpenAI

use_kimi = False #True
N = 10

if use_kimi:
    client = OpenAI(
        api_key="Y2w0dXFxMXI2a2plaXVudDFhdDA6bXNrLWF0RGxuUWllNjhmME9lZTJJcWtwYnRkbDE1bEo=",
        base_url="https://api.moonshot.cn/v1")
else:
    client = OpenAI(
        api_key="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",
    )

with open('./data/parallel_test.json', encoding="utf-8") as file_obj:
    dataset = json.load(file_obj)

def cal(filename, tot_num=100):
    with open(filename, encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    res1 = 0
    res2 = 0
    scored_num = 0
    for idx in range(tot_num):
        if idx % 20 == 0:
            print("small break")
            time.sleep(10)
            print("we are back!")
        en = dataset[idx]['en']
        zh = data[idx]
        response = client.chat.completions.create(
            model="moonshot-v1-8k" if use_kimi else "gpt-3.5-turbo-1106",
            messages=[
            {"role": "system", "content": "You should strictly follow the user's requirements."},
            {"role": "user", "content": f'''You are a translation grader. Given English lyrics and a corresponding Chinese translation, you need to give scores in range 1-4 (4 is the highest) for fluency and translation accuracy respectively. 

        Here are the metrics for fluency (you should only consider the Chinese sentence here, no need to consider whether the translation is correct):
        Score 1: Not fluent, can't understand the meaning easily.
        Score 2: There are inappropriate or awkward phrases or other big unbearable flaws.
        Score 3: Quite fluent. There exists small mistakes, but they are acceptable.
        Score 4: Very fluent and no mistakes.

        Here are the metrics for translation accuracy:
        Score 1: More than 50% is translated wrongly, or there are unbearable translation mistakes.
        Score 2: Merely acceptable, but there are mistakes that need correction.
        Score 3: No big mistake in translation, totally acceptable. But there is still space for improvement, such as phrasing or the translation of idioms.
        Score 4: Excellent translation.

        Now, I will provide you with the English lyrics and the Chinese translation. You need to give me only two numbers(first for fluency and second for accuracy) and nothing else. 
        The English lyrics is: {en}
        The Chinese translation is: {zh}
        '''}],
            n=N,
            temperature=0.7,
            max_tokens=500,
        )
        time.sleep(2)
        
        print([response.choices[i].message.content for i in range(N)], response.usage.total_tokens)
        r1 = 0
        r2 = 0
        num = 0
        for i in range(N):
            s = [float(n) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', response.choices[i].message.content)]
            if len(s) >= 2:
                score1 = s[0]
                score2 = s[1]
                r1 += score1
                r2 += score2
                num += 1
        if num > 0:
            r1 /= num   
            r2 /= num
            res1 += r1
            res2 += r2
            scored_num += 1
        print("en=", en, "zh=", zh, "fluency=", r1, "translation=", r2)
        print("---------------------------", idx, " / ", tot_num, "finished-----------------------------------")
    print(filename, "results are:", res1 / scored_num, res2 / scored_num)

# cal('original_gen_res.json')
# cal('SFT_gen_res.json')
# cal('DPOv1_gen_res.json')
# cal('DPOv3dot5_gen_res.json')
# cal('SFTv2_gen_res.json')
# cal('SFTv3_gen_res.json')
# cal('DPOv4dot1_gen_res.json')
cal('DPOv7dot9_gen_res.json')