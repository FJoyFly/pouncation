import tensorflow as tf
import numpy as np
import heapq
import copy
import re

# A = [[[1.0, 3.0, 4.0, 9.0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],

b = [[[], [1, 2, 3], [1, 2, 3]],
     [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
     [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
     ]
FFF = 'M M M M M M M M '
f = FFF.replace(' ', '')
print(f)
n = [1, 2, 3]
if n is None:
    print('yes')
else:
    print('No')
a = 3
b = 7
print(a / b)
uhu = format(3 / 7, '0.3f')
print(uhu)
# for i in range(3):
#     now_score_nine[i] += score_three[i]
#     now_score_nine[i + 3] += score_three[i + 1]
#     now_score_nine[i + 6] += score_three[i + 2]
