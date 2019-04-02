import re
from itertools import chain
from os import makedirs
from os.path import exists, join
import pickle
import numpy as np
import pandas as pd
from keras.utils import np_utils
import codecs

train_input_file_path = '/home/joyfly/桌面/副本2'


# 取出格式化的数据并返回一个大列表
def get_data(input_file):
    '''

    :param input_file: 终整理文本输入
    :return: 返回一个所需的list
    '''
    all_word_list = list()
    input_data = codecs.open(input_file, 'r', 'utf-8')
    for line in input_data.readlines():
        word_list = line.strip().split()
        all_word_list.append(word_list)
    input_data.close()
    return all_word_list


# ?为什么最后一个字不见了
def get_data_label(word):
    """
    针对于utf-8的编码。每个段尾可能时以\0\0结尾的
    在正则匹配的时候最后一个元素不能正常匹配到
    增加一个空格符号在最后，以便于re模块更准确的匹配
    :param word:
    :return: 返回汉字和标签对应的list
    """
    word.extend(' ')
    word = u' '.join(word)
    word = re.findall('(.)/([BMSE].)', word)
    if word:
        word = np.array(word)
        return list(word[:, 0]), list(word[:, 1])


data = []
label = []

# 获取数据集
all_list_word = get_data(train_input_file_path)

# 获取训练集中古文和标签对应的集合
for duan in all_list_word:
    da = get_data_label(duan)
    if da:
        data.append(da[0])
        label.append(da[1])
print('data length ', len(data), 'Label length ', len(label))
print('data example', data[0], )
print('label example', label[0])

maxlen = 256  # 此处的maxlen设置的是每段古文默认的字数，若超过这个字数，则采取截断措施，不够则padding 0

# 需要设置文字和标注转换化为index,根据index转换为文字或者标注
all_words = list(chain(*data))
all_data_sr = pd.Series(all_words)
all_data_counts = all_data_sr.value_counts()
all_data_set = all_data_counts.index
all_data_ids = range(1, len(all_data_set) + 1)

# 字转化id,id转化字
word2id = pd.Series(all_data_ids, index=all_data_set)
print('word2id:\n', word2id)
id2word = pd.Series(all_data_set, index=all_data_ids)
print('id2word:\n', id2word)
tags_set = ['S ', 'B ', 'M ', 'E3', 'E2', 'E ', 'X ']
tags_ids = range(len(tags_set))

# 标注转化id,id转化标注
tag2id = pd.Series(tags_ids, index=tags_set)
id2tag = pd.Series(tags_set, index=tags_ids)


def word_trans(word):
    ids = list(word2id[word])
    if len(ids) >= maxlen:
        ids = ids[:maxlen]
    ids.extend([0] * (maxlen - len(ids)))
    return ids


def tag_trans(tag):
    ids = list(tag2id[tag])
    if len(ids) >= maxlen:
        ids = ids[:maxlen]
    ids = np_utils.to_categorical(ids, 7)
    ids = list(ids)
    for _ in range(maxlen - len(ids)):
        ids.extend([np.array([0, 0, 0, 0, 0, 0, 1])])
    ids.extend([np.array([0, 0, 0, 0, 0, 0, 1])] * (maxlen - len(ids)))
    return np.array(ids)


data_word = list(map(lambda x: word_trans(x), data))
data_tag = list(map(lambda y: tag_trans(y), label))

print(" 汉字以及标注转换完成 \n")
data_word = np.asarray(data_word)
data_tag = np.asarray(data_tag)
print("Starting pickle to file.....\n")

path = '/home/joyfly/桌面'
if not exists(path):
    makedirs(path)

with open(join(path, "all_data.pkl"), 'wb') as f:
    pickle.dump(data_word, f)
    pickle.dump(data_tag, f)
    pickle.dump(word2id, f)
    pickle.dump(id2word, f)
    pickle.dump(tag2id, f)
    pickle.dump(id2tag, f)
print('Pickle Finished!')
