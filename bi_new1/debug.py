import tensorflow as tf
import numpy as np
import heapq
import copy
import re
import os.path
import shutil
import matplotlib.pyplot as plt
import pandas as pd

# import codecs
#
# input_data = codecs.open('/home/joyfly/桌面/兰亭集序')
# for line in input_data.readlines():
#     line = line.strip().split()
#     for i, word in enumerate(line):
#         if u'\u4E00' <= word <= u'\u9FEF':
#             # fuhao = re.match('[，。？！：；、]', line[i + 1])
#             if line[i + 1] == '，':
#                 tag = 'j'
#                 print(word + "/S" + tag)


tag = 'B'
if tag.isupper():
    print('true')
# for i in range(3):
#     now_score_nine[i] += score_three[i]
#     now_score_nine[i + 3] += score_three[i + 1]
#     now_score_nine[i + 6] += score_three[i + 2]
