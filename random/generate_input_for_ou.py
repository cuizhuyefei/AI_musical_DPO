import os
import numpy as np
import json
import re
    
# read in song_lyrics.json
song_lyrics = []
with open('final_experiments/song_lyrics.json', 'r') as f:
    song_lyrics = json.load(f)
all_paragraphs = []
for song in song_lyrics:
    paragraphs = re.split(r'\n\s*\n', song['lyrics'])
    # paragraphs[0] = paragraphs[0][1:] # remove the first '\n'
    paragraphs[-1] = paragraphs[-1][:-1] # remove the last '\n'
    all_paragraphs.extend(paragraphs)
    
num_lines = 0
for par in all_paragraphs:
    num_lines += len(par.split('\n'))
print(f"Number of paragraphs: {len(all_paragraphs)}")
print(f"Number of lines: {num_lines}")

# output all paragraphs to a txt file
with open('all_paragraphs.txt', 'w') as f:
    for par in all_paragraphs:
        f.write(par)
        f.write('\n\n')