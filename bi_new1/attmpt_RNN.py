import time
import numpy as np
import random
import os
import re
import codecs

embedding_size = 256  # 隐层节点数
layer_num = 2  # bi-GRU层数
max_grad = 5.0  # 最大梯度
train_input_file_path = '/home/joyfly/桌面/副本'
epochs = 200000
pouncation_num = 6
batch_size = 256
window_size = 5
num_epoch_improvement = 10
dir = '/home/joyfly/桌面/session'



# 初始化
def __init__(self, embedding, layer_num, max_grad, vocab_size, pouncation_num, batch_size, window_size):
    self.embedding = embedding
    self.layer_num = layer_num
    self.max_grad = max_grad
    self.vocab_size = vocab_size
    self.pouncation_num = pouncation_num
    self.batch_size = batch_size
    self.window_size = window_size


# 初始化sess 并保存模型参数
def initialize_session():
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)


# 恢复session中参数
def restore_sess(self, dir_path):
    if dir_path is None:
        dir_path = dir
    self.saver.restore(self.sess, dir_path)


# 关闭session
def close_sess(self):
    self.sess.close()


# 初始化权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial)


# 初始化各偏置值
def bias_variavle(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial)


#  双向GRU字向量和GRU各参数进行预训练


# 定义损失函数
def compute_cross(self, inputs, labels):
    self.pre_labels_logits = self.bi_gru(inputs)
    self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(self.pre_labels_logits), reduction_indices=[1]))
    return self.cross_entropy


# 定义梯度下降
def Adamoptimizer_down(self, inputs, labels):
    self.train_step = tf.train.AdamOptimizer(learing_rate).minimize(compute_cross(inputs, labels))
    return self.train_step



# 测试验证集__
def prediction(self, test):
    """
    验证集和测试集运行，并统计结果，采用精确率 (P) 、召回率 (R) 、 F1 值以及准
确率 (A)进行评估模型
    """
    correction_poun, error_poun, error_nopoun, correction_nopoun = 0., 0., 0., 0.
    for inputs, labels in generate_minibatch(test, batch_size, window_size):
        pre_labels = pre_tag(inputs)
        abels_pre = tf.cast(tf.argmax(pre_labels, axis=-1), tf.int32)
        abels_real = tf.cast(tf.argmax(labels, axis=-1), tf.int32)
        if (abels_pre == 0 or abels_pre == 5) and (abels_real == 0 or abels_real == 5):
            correction_poun += 1
        elif (abels_pre != 0 and abels_pre != 5) and (abels_real == 0 or abels_real == 5):
            error_poun += 1
        elif (abels_pre == 0 or abels_pre == 5) and (abels_real != 0 and abels_real != 5):
            error_nopoun += 1
        else:
            correction_nopoun += 1
    P = correction_poun / (correction_poun + error_poun) if correction_poun > 0 else 0
    R = correction_poun / (correction_poun + error_nopoun) if correction_poun > 0 else 0
    F1 = 2 * P * R / (P + R)
    A = (correction_poun + correction_nopoun) / (correction_poun + correction_nopoun + error_nopoun + error_poun)
    return F1, A



