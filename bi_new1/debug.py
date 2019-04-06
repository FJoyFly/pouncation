import tensorflow as tf
import numpy as np
import heapq
import copy
import re
import os.path
import shutil
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [9, 8, 7, 6]
q = [1, 2, 3, 4]
plt.plot(q, y)
plt.scatter(q, y)
plt.plot(q, x)
plt.scatter(q, x)
plt.savefig('/home/joyfly/桌面/image.png')
num_classes = 10
# 需要转换的整数
arr = [1, 3, 4, 5, 9]
# 将整数转为一个10位的one hot编码
print(np.eye(10)[arr])

# for i in range(3):
#     now_score_nine[i] += score_three[i]
#     now_score_nine[i + 3] += score_three[i + 1]
#     now_score_nine[i + 6] += score_three[i + 2]
