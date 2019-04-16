import tensorflow as tf
import pickle
import math
import numpy as np
from sklearn.model_selection import train_test_split
from os.path import join
# import heapq
import os
import codecs
import shutil
import matplotlib.pyplot as plt
import matplotlib

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

source_data = '/home/joyfly/桌面/all_data.pkl'
train_batch_size = 128
dev_batch_size = 128
test_batch_size = 128
embedding_size = 256
batch_size = 128
layer_num = 2
pouncation_num = 12
learning_rate = 0.001
isTrain = False
epochs = 500
summaries_dir = '/home/joyfly/桌面/summary/'
save_dir = '/home/joyfly/桌面/ckpt/'
outputs_path = '/home/joyfly/桌面/'
steps_per_print = 6
steps_per_sumary = 5
epochs_per_dev = 2
num_epoch_no_improve_bear = 10
sequence_length = 256
lamda = 0.7
zhfont1 = matplotlib.font_manager.FontProperties(fname="/home/joyfly/下载/SimHei.ttf")


def load_data():
    """
    载入数据from pickle
    :return: Arrays
    """
    with open(source_data, 'rb') as f:
        data_word = pickle.load(f)
        data_tag = pickle.load(f)
        word2id = pickle.load(f)
        id2word = pickle.load(f)
        tag2id = pickle.load(f)
        id2tag = pickle.load(f)
        return data_word, data_tag, word2id, id2word, tag2id, id2tag


def get_data(data_word, data_label):
    """
    熊数据集中分出训练集以及开发集和测试集
    :param data_word: 输入字
    :param data_label: 输入标签
    :return: Arrays
    """

    train_word, test_word, train_label, test_label = train_test_split(data_word, data_label, test_size=0.2,
                                                                      random_state=50)
    train_word, dev_word, train_label, dev_label = train_test_split(train_word, train_label, test_size=0.2,
                                                                    random_state=50)
    return train_word, train_label, dev_word, dev_label, test_word, test_label


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(value=0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(initial)


def judge(x):
    if -1 < x < 11:
        return x


# 结果评估
def compute_PRFvalues(label_1, label_2):
    """

    :param label_1: 预测标签
    :param label_2: 真实标签
    :return: 返回对应的P值，R值，F值以及准确率
    """
    real_poun = 0
    fail_poun = 0
    fail_nopoun = 0
    real_nopoun = 0

    for duan1, duan2 in zip(label_1, label_2):
        for i in range(len(duan1)):
            if -1 < duan2[i] < 6 and -1 < duan1[i] < 6:
                if duan1[i] < 2 and duan2[i] < 2:
                    real_poun += 1
                elif duan1[i] < 2 and duan2[i] > 1:
                    fail_poun += 1
                elif duan1[i] > 1 and duan2[i] < 2:
                    fail_nopoun += 1
                elif duan1[i] > 1 and duan2[i] > 1:
                    real_nopoun += 1
            else:
                break
    P_value = round(real_poun / (real_poun + fail_poun), 3)
    R_value = round(real_poun / (real_poun + fail_nopoun), 3)
    F_value = round(2 * P_value * R_value / (P_value + R_value), 3)
    A_value = round((real_nopoun + real_poun) / (real_nopoun + real_poun + fail_nopoun + fail_poun), 3)
    return P_value, R_value, F_value, A_value


def new_compute_evaluate(lable1, lable2):
    '''

    :param lable1:所得预测标签
    :param lable2: 真是标签
    :return:
    '''
    real_poun = 0
    fail_poun = 0
    fail_nopoun = 0
    real_nopoun = 0
    real_biao = 0
    pre_biao = 0
    all_biao = 0
    for duan1, duan2 in zip(lable1, lable2):
        for i in range(len(duan1)):
            if duan1[i].isupper() and duan2[i].isupper():
                real_nopoun += 1
            elif duan1[i].islower() and duan2[i].islower():
                if duan1[i] == duan2[i]:
                    real_biao += 1
                real_poun += 1
            elif duan1[i].isupper() and duan2[i].islower():
                fail_poun += 1
            elif duan1[i].islower() and duan2[i].isupper():
                fail_nopoun += 1
            if duan1[i].islower():
                pre_biao += 1
            if duan2[i].islower():
                all_biao += 1
    print(real_biao)
    print(pre_biao)
    biaodian_P = round(real_biao / pre_biao, 3)
    biaodian_R = round(real_biao / all_biao, 3)
    biaodian_F = round(2 * biaodian_P * biaodian_R / (biaodian_P + biaodian_R), 3)
    duanju_P_value = round(real_poun / (real_poun + fail_poun), 3)
    duanju_R_value = round(real_poun / (real_poun + fail_nopoun), 3)
    duanju_F_value = round(2 * duanju_P_value * duanju_R_value / (duanju_P_value + duanju_R_value), 3)
    duanju_A_value = round((real_nopoun + real_poun) / (real_nopoun + real_poun + fail_nopoun + fail_poun), 3)
    return duanju_P_value, duanju_R_value, duanju_F_value, duanju_A_value, biaodian_P, biaodian_R, biaodian_F


# 显示对比解结果
def display_word(word, label, file):
    '''

    :param word: 一段古文
    :param label1: 预测该段古文的标签
    :param label2: 该段古文的真实标签
    :return: 打印出对比结果
    '''
    outputdata = codecs.open(join(outputs_path, file), 'a+', 'utf-8')
    word_list = word.split(' ')
    outputdata.write('  ')
    for i in range(len(word_list)):
        if label[i] == 'd':
            outputdata.write(word_list[i] + '，' + ' ')
        elif label[i] == 'j':
            outputdata.write(word_list[i] + '。' + ' ')
        elif label[i] == 'w':
            outputdata.write(word_list[i] + '？' + ' ')
        elif label[i] == 'g':
            outputdata.write(word_list[i] + '！' + ' ')
        elif label[i] == 'f':
            outputdata.write(word_list[i] + '；' + ' ')
        elif label[i] == 'm':
            outputdata.write(word_list[i] + '：' + ' ')
        elif label[i] == 't':
            outputdata.write(word_list[i] + '、' + ' ')
        else:
            outputdata.write(word_list[i])

    outputdata.write('\n')
    outputdata.close()


def dele_none(label):
    for i in range(len(label)):
        if label[len(label) - 1] is None:
            label.pop()
        else:
            break
    for i in range(len(label)):
        if label[i] is None:
            label[i] = 2
    return label


def main():
    global learning_rate, three_batchsize, flag
    # 加载数据
    data_word, data_label, word2id, id2word, tag2id, id2tag = load_data()
    # 分割数据集
    train_word, train_label, dev_word, dev_label, test_word, test_label = get_data(data_word, data_label)
    train_step = math.floor(train_word.shape[0] / train_batch_size)
    # train_restseq_num = train_word.shape[0] - train_batch_size * (train_step - 1)
    dev_step = math.floor(dev_word.shape[0] / dev_batch_size)
    # dev_restseq_num = dev_word.shape[0] - dev_batch_size * (dev_step - 1)
    test_step = math.floor(test_word.shape[0] / test_batch_size)
    # test_restseq_num = test_word.shape[0] - test_batch_size * (test_step - 1)
    vocab_size = len(word2id) + 1

    global_step = tf.Variable(-1, trainable=False, name='global_step')

    # 各数据集batch
    train_dataset = tf.data.Dataset.from_tensor_slices((train_word, train_label))
    train_dataset = train_dataset.batch(train_batch_size, drop_remainder=True)

    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_word, dev_label))
    dev_dataset = dev_dataset.batch(dev_batch_size, drop_remainder=True)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_word, test_label))
    test_dataset = test_dataset.batch(test_batch_size, drop_remainder=True)

    """
    这里有3中常用迭代器，one hot iterator:这是最简单的迭代器
                     可初始化的迭代器：initializable_iterator
                     可重新初始化的迭代器：转换数据集见下,这种数据集每次更换iterator时,数据会重头开始
    """

    # 构造一个通用迭代器，并且对训练集，验证集，测试集初始化
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

    train_initial_op = iterator.make_initializer(train_dataset)
    dev_initial_op = iterator.make_initializer(dev_dataset)
    test_initial_op = iterator.make_initializer(test_dataset)

    # Input layer
    data_word_x, data_label_y = iterator.get_next()

    # if flag:
    #     seq = np.full(batch_size, sequence_length, dtype=np.int32)
    # else:
    #     seq = np.full(three_batchsize, sequence_length, dtype=np.int32)
    # seq = np.linspace(256, 256, 256)
    seq = np.full(batch_size, sequence_length, dtype=np.int32)
    # embedding layer
    embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1), dtype=tf.float32)
    inputs = tf.nn.embedding_lookup(embeddings, data_word_x)

    # 梯度下降
    # keep_prob = tf.placeholder(tf.float32, [])

    # 采用双向GRU的Rnn
    gru_fw_cell = tf.nn.rnn_cell.GRUCell(num_units=embedding_size)
    gru_lw_cell = tf.nn.rnn_cell.GRUCell(num_units=embedding_size)

    # 为避免过拟合使用dropout函数
    gru_fw_cell = tf.nn.rnn_cell.DropoutWrapper(gru_fw_cell, input_keep_prob=1.0, output_keep_prob=1.0)
    gru_lw_cell = tf.nn.rnn_cell.DropoutWrapper(gru_lw_cell, input_keep_prob=1.0, output_keep_prob=1.0)

    # 设置多层GRU
    stacked_fw_cell = tf.nn.rnn_cell.MultiRNNCell([gru_fw_cell] * layer_num, state_is_tuple=True)
    stacked_lw_cell = tf.nn.rnn_cell.MultiRNNCell([gru_lw_cell] * layer_num, state_is_tuple=True)

    # 初始化状态
    # initial_fw_cell = stacked_fw_cell.zero_state()
    # initial_lw_cell = stacked_lw_cell.zero_state()

    inputs = tf.unstack(inputs, axis=1)
    outputs, _, _ = tf.nn.static_bidirectional_rnn(stacked_fw_cell, stacked_lw_cell, inputs, dtype=tf.float32,
                                                   sequence_length=seq)
    output = tf.stack(outputs, axis=1)
    output = tf.reshape(tf.concat(output, 1), [-1, embedding_size * 2])

    # output layer
    with tf.name_scope('weight'):
        softmax_weight = weight_variable([embedding_size * 2, pouncation_num])
        tf.summary.histogram('weight', softmax_weight)
    with tf.name_scope('bias'):
        softmax_bias = bias_variable([pouncation_num])
        tf.summary.histogram('bias', softmax_bias)
    begin_pre_labels = tf.matmul(output, softmax_weight) + softmax_bias
    begin_pre_labels_reshape = tf.reshape(begin_pre_labels, (-1, sequence_length, 12))

    # 原始每段句子的标注结果标签概率

    pre_labels_reshape = tf.nn.softmax(begin_pre_labels_reshape, axis=-1)

    y_label_pre = tf.cast(tf.argmax(pre_labels_reshape, axis=-1), tf.int32)

    # 真实标签
    y_label_reshape = tf.reshape(data_label_y, (-1, sequence_length, 12))

    y_label_real = tf.cast(tf.argmax(y_label_reshape, axis=-1), tf.int32)

    # CRF层
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
        begin_pre_labels_reshape, y_label_real, seq)

    viterbi_sequence, _ = tf.contrib.crf.crf_decode(begin_pre_labels_reshape, transition_params, seq)
    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(-log_likelihood)
        tf.summary.scalar('loss', cross_entropy)

    # pre_labels_reshape = tf.nn.softmax(viterbi_sequence, axis=-1)
    # y_label_pre = tf.cast(tf.argmax(pre_labels_reshape, axis=-1), tf.int32)

    # prediction
    correct_prediction = tf.equal(y_label_pre, y_label_real)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    print('Prediction', correct_prediction, 'Accuracy', accuracy)

    # loss softmax的损失函数
    # with tf.name_scope('loss'):
    #     cross_entropy = tf.reduce_mean(
    #         tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label_real,
    #                                                        logits=begin_pre_labels_reshape + 1e-10))
    #     tf.summary.scalar('loss', cross_entropy)

    # train控制梯度
    # optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    # grads = optimizer.compute_gradients(cross_entropy)
    # for i, (g, v) in enumerate(grads):
    #     if g is not None:
    #         grads[i] = (tf.clip_by_norm(g, 5), v)  # 阈值这里设为5
    # train_step_op = optimizer.apply_gradients(grads, global_step)
    train_step_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step)

    # Saver
    saver = tf.train.Saver(max_to_keep=1)

    sess = tf.InteractiveSession()

    # config = tf.ConfigProto(device_count={"CPU": 12}, inter_op_parallelism_threads=0, intra_op_parallelism_threads=0,
    #                         )

    sess.run(tf.global_variables_initializer())

    # summary
    if isTrain:
        if os.path.exists(join(summaries_dir, 'train')):
            shutil.rmtree(join(summaries_dir, 'train'))
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(join(summaries_dir, 'train'), sess.graph)
    else:
        if os.path.exists(join(summaries_dir, 'test')):
            shutil.rmtree(join(summaries_dir, 'test'))
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(join(summaries_dir, 'test'), sess.graph)

    gstep = 0
    Flag = 0

    best_score = 0
    num_epoch_no_improve = 0
    # feed sequence

    # ---------------------------------------
    # 名字不能重复使用 尽量避免因同名带来的危险
    # ---------------------------------------
    if isTrain:
        print("Traing......")
        for epoch in range(epochs):
            print('当前轮次为：', epoch)
            tf.train.global_step(sess, global_step_tensor=global_step)
            sess.run(train_initial_op, feed_dict={})
            for step in range(int(train_step)):
                summary_run, labels_pre_h, loss, acc, gstep, _ = sess.run(
                    [summaries, begin_pre_labels_reshape, cross_entropy, accuracy, global_step,
                     train_step_op])

                # smrs, loss, acc, gstep, _ = sess.run([summaries, cross_entropy, accuracy, global_step, train_step_op],
                #                                      feed_dict={keep_prob: 1})
                print("当前batch:", step, 'loss=', loss, 'accuracy=', acc)
                if step % steps_per_print == 0:
                    print('Global Step', gstep, 'Step', step, 'Trian loss', loss, 'Train Accuracy', acc)

                if gstep % steps_per_sumary == 0:
                    writer.add_summary(summary_run, gstep)
                    print('Write Summaries to', summaries_dir)

            if epoch % epochs_per_dev == 0:
                print('正在验证中：')
                sess.run(dev_initial_op)
                print('sessing')
                for step in range(int(dev_step)):
                    print('step', step)
                    # 此处使用的minibatch梯度下降,将数据集分成train_step份.在每一次小batch中更新参数
                    if step % steps_per_print == 0:
                        acc = sess.run(accuracy)

                        print('running')
                        if acc > best_score:
                            num_epoch_no_improve = 0
                            best_score = acc
                            saver.save(sess, join(save_dir, 'model.ckpt'), global_step=gstep, write_meta_graph=False)
                            print('Dev Accuracy', acc, 'Step', step)
                        else:
                            num_epoch_no_improve += 1
                            learning_rate = learning_rate * lamda
                            if num_epoch_no_improve > num_epoch_no_improve_bear:
                                Flag = 1
                                break
            if Flag:
                break
    else:
        number_print = 0
        # final_label_pre_result = []
        print("Testing......")
        # load model
        last_ckpt = tf.train.latest_checkpoint(save_dir)
        if last_ckpt:
            saver.restore(sess, last_ckpt)
        print('Restore From', last_ckpt)
        sess.run(test_initial_op)
        P, R, F, A, P_b, R_b, F_b = [], [], [], [], [], [], []
        for step in range(int(test_step)):
            summary_run, data_word_result, data_label_pre_result, data_label_real_result, acc = sess.run(
                [summaries, data_word_x, viterbi_sequence, y_label_real, accuracy])

            print("test step", step, '未处理前Accuracy', acc)
            if gstep % steps_per_sumary == 0:
                writer.add_summary(summary_run, gstep)
                print('Write Summaries to', summaries_dir)
            com1 = []
            com2 = []
            for i in range(len(data_word_result)):
                y_real_label_ = list(map(judge, data_label_real_result[i]))
                y_real_label_ = dele_none(y_real_label_)
                data_word_result_final = list(filter(lambda x: x, data_word_result[i]))
                y_predict_label_final = list(map(judge, data_label_pre_result[i]))
                y_predict_label_final = dele_none(y_predict_label_final)
                word_deal = ' '.join(id2word[data_word_result_final].values)
                list_label_y = id2tag[y_predict_label_final].values
                list_label_y_real = id2tag[y_real_label_].values
                if number_print < 10:
                    if len(list(word_deal)) > 150:
                        print(word_deal)
                        print(list_label_y)
                        print(list_label_y_real)
                        display_word(word_deal, list_label_y, '预测')
                        display_word(word_deal, list_label_y_real, '原始')
                        number_print += 1
                com1.append(list_label_y)
                com2.append(list_label_y_real)
                # word_x = ''.join(id2word[data_word_result_final].values)
                # label_y = ''.join(id2tag[y_predict_label_final].values)
                # label_y_real = ''.join(id2tag[y_real_label_].values)
                # label_y = label_y.replace(' ', '')
                # label_y_real = label_y_real.replace(' ', '')
                # print(word_x)
                # print(label_y)
                # print(label_y_real)
            m1, m2, m3, m4, m5, m6, m7 = new_compute_evaluate(com1, com2)
            print(m1, m2, m3, m4)
            print(m5, m6, m7)
            P.append(m1)
            R.append(m2)
            F.append(m3)
            A.append(m4)
            P_b.append(m5)
            R_b.append(m6)
            F_b.append(m7)
            # new_compute_evaluate(com1, com2)
        X_pan = np.linspace(0, int(test_step) - 1, int(test_step))
        plt.plot(X_pan, P_b)
        plt.scatter(X_pan, P_b)
        plt.plot(X_pan, R_b)
        plt.scatter(X_pan, R_b)
        plt.plot(X_pan, F_b)
        plt.scatter(X_pan, F_b)
        plt.title('the value of P,R,F in the biao_test')
        plt.xlabel('test_step')
        plt.ylabel('P,R,F')
        plt.savefig('/home/joyfly/桌面/image.png')
        plt.close()
        plt.plot(X_pan, P)
        plt.scatter(X_pan, P)
        plt.plot(X_pan, R)
        plt.scatter(X_pan, R)
        plt.plot(X_pan, F)
        plt.scatter(X_pan, F)
        plt.plot(X_pan, A)
        plt.scatter(X_pan, A)
        plt.title('the value of P,R,F in the duan_test')
        plt.xlabel('test_step')
        plt.ylabel('P,R,F,A')
        plt.savefig('/home/joyfly/桌面/image1.png')
        plt.close()


if __name__ == '__main__':
    main()
