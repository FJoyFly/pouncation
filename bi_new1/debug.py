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
tag1 = 'B'
if tag == tag1:
    print('true')

list1 = [1, 2, 3, 4, 5]
list2 = [6, 7, 8, 9, 5]
plt.plot(list1, list2)
plt.savefig('/home/joyfly/桌面/im.png')
plt.close()
plt.plot(list2, list1)
plt.savefig('/home/joyfly/桌面/im1.png')
plt.close()
