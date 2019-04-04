import tensorflow as tf
import pickle
import codecs
from test import search_bestresult, judge, dele_none
import numpy as np

sequence_maxlen = 256

path = '/home/joyfly/桌面/inputs_data'
source_data = '/home/joyfly/桌面/word2id_data.pkl'
ckpt_path = '/home/joyfly/桌面/ckpt/'
maxlen = 256


def load_data():
    """
    载入数据from pickle
    :return: Arrays
    """
    with open(source_data, 'rb') as f:
        tag2id = pickle.load(f)
        id2tag = pickle.load(f)
        word2id = pickle.load(f)
        id2word = pickle.load(f)
        return tag2id, id2tag, word2id, id2word


tag2id, id2tag, word2id, id2word = load_data()
print(word2id)


def word_trans(word):
    ids = list(word2id[word])
    if len(ids) >= maxlen:
        ids = ids[:maxlen]
    ids.extend([0] * (maxlen - len(ids)))
    return ids


inputs_data = codecs.open(path, 'r', 'utf-8')
new_data = []
for line in inputs_data:
    if 256 < len(line) <= 512:
        middle_num = len(line) // 2
        for i in range(middle_num):
            if line[middle_num + i].isspace():
                line1 = line[:middle_num + i]
                line2 = line[middle_num + i + 1:]
                new_data.append(line1)
                new_data.append(line2)
                break
    elif len(line) <= 256:
        new_data.append(line)

data_list = []

for line in new_data:
    line_data = []
    line = line.strip().split()
    for word in line:
        for i in word:
            if u'\u4E00' <= i <= u'\u9FEF':
                line_data.extend(i)
    data_list.append(line_data)

data_word = list(map(lambda x: word_trans(x), data_list))

for i in data_word:
    print(i)
test_word = tf.data.Dataset.from_tensor_slices(data_word)
iterator = test_word.make_one_shot_iterator()

try:
    inputs = iterator.get_next()
    list = inputs.get_shape().as_list()
except tf.errors.OutOfRangeError:
    print('已经取完')

sess = tf.Session()

saver = tf.train.import_meta_graph(ckpt_path + 'model.ckpt-517.meta')
saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))

graph = tf.get_default_graph()

keep_prob = graph.get_tensor_by_name('keep_prob:0')
begin_pre_labels_reshape = graph.get_tensor_by_name('begin_pre_labels_reshape:0')

sess.run(iterator)

print('迭代器完成')

data_label_pre_result, data_word_result = sess.run([begin_pre_labels_reshape, inputs], feed_dict={keep_prob: 1})

print('得到标注标签')
data_label_pre_result = search_bestresult(data_label_pre_result)
for i in range(len(data_label_pre_result)):
    data_word_result_final = list(filter(lambda x: x, data_word_result[i]))
    data_label_result_final = list(map(judge, data_label_pre_result[i]))
    y_predict_label_final = dele_none(data_label_result_final)
    word_x = ''.join(id2word[data_word_result_final].values)
    label_y = ''.join(id2tag[y_predict_label_final].values)
    print(word_x, label_y)
