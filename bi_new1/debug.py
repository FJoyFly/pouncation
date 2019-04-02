import tensorflow as tf
import numpy as np
import heapq
import copy
import re
import os.path
import shutil

path = '/home/joyfly/桌面/summary/test'
if os.path.exists(path):
    shutil.rmtree(path)
    print('yes')
else:
    print('No')

# for i in range(3):
#     now_score_nine[i] += score_three[i]
#     now_score_nine[i + 3] += score_three[i + 1]
#     now_score_nine[i + 6] += score_three[i + 2]
