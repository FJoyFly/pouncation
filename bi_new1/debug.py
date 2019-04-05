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

# for i in range(3):
#     now_score_nine[i] += score_three[i]
#     now_score_nine[i + 3] += score_three[i + 1]
#     now_score_nine[i + 6] += score_three[i + 2]
