import tensorflow as tf
import numpy as np
import heapq
import copy
import re
import os.path
import shutil

batchsize_num = tf.placeholder(tf.int32, [])
seq = np.full(batchsize_num, 2, dtype=np.int32)

with tf.Session() as sess:
    print(sess.run([seq], feed_dict={batchsize_num: 3}))
# for i in range(3):
#     now_score_nine[i] += score_three[i]
#     now_score_nine[i + 3] += score_three[i + 1]
#     now_score_nine[i + 6] += score_three[i + 2]
