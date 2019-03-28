import tensorflow as tf
import pickle
import math
import numpy as np
from sklearn.model_selection import train_test_split
from os.path import join

source_data = '/home/joyfly/桌面/all_data.pkl'
train_batch_size = 256
dev_batch_size = 256
test_batch_size = 256
embedding_size = 256
layer_num = 2
batch_size = 512
pouncation_num = 7
learning_rate = 0.8
isTrain = True
epochs = 5000
summaries_dir = '/home/joyfly/桌面/summary'
save_dir = '/home/joyfly/桌面/ckpt/'
steps_per_print = 300
steps_per_sumary = 500
epochs_per_dev = 2
num_epoch_no_improve_bear = 10
sequence_length = 512


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


def search_bestresult(waiting_deal_label):
    score = np.zeros(3, tf.float32)
    label = [3]
    for duan in waiting_deal_label:
        for zi in duan:
            break


def main():
    global learning_rate
    # 加载数据
    data_word, data_label, word2id, id2word, tag2id, id2tag = load_data()

    # 分割数据集
    train_word, train_label, dev_word, dev_label, test_word, test_label = get_data(data_word, data_label)
    train_step = math.ceil(train_word.shape[0] / train_batch_size)
    dev_step = math.ceil(dev_word.shape[0] / dev_batch_size)
    test_step = math.ceil(test_word.shape[0] / test_batch_size)
    vocab_size = len(word2id) + 1
    print("vocabulary size:", vocab_size)

    global_step = tf.Variable(-1, trainable=False, name='global_step')

    # 各数据集batch
    train_dataset = tf.data.Dataset.from_tensor_slices((train_word, train_label))
    train_dataset = train_dataset.batch(train_batch_size)
    print(train_dataset)

    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_word, dev_label))
    dev_dataset = dev_dataset.batch(dev_batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_word, test_label))
    test_dataset = test_dataset.batch(test_batch_size)

    """
    这里有3中常用迭代器，one hsot iterator:这是最简单的迭代器
                     可初始化的迭代器：initializable_iterator
                     可重新初始化的迭代器：转换数据集见下
    """

    # 构造一个通用迭代器，并且对训练集，验证集，测试集初始化
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

    train_initial_op = iterator.make_initializer(train_dataset)
    dev_initial_op = iterator.make_initializer(dev_dataset)
    test_initial_op = iterator.make_initializer(test_dataset)

    # Input layer
    data_word, data_label = iterator.get_next()

    # embedding layer
    embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1), dtype=tf.float32)
    inputs = tf.nn.embedding_lookup(embeddings, data_word)
    print(inputs.shape)

    # 梯度下降
    keep_prob = tf.placeholder(tf.float32, [])

    # 采用双向GRU的Rnn
    gru_fw_cell = tf.nn.rnn_cell.GRUCell(num_units=embedding_size)
    gru_lw_cell = tf.nn.rnn_cell.GRUCell(num_units=embedding_size)

    # 对dropout函数初始化
    drop_keep_rate = tf.placeholder(tf.float32, name="dropout_keep")

    # 为避免过拟合使用dropout函数
    gru_fw_cell = tf.nn.rnn_cell.DropoutWrapper(gru_fw_cell, input_keep_prob=1.0, output_keep_prob=drop_keep_rate)
    gru_lw_cell = tf.nn.rnn_cell.DropoutWrapper(gru_lw_cell, input_keep_prob=1.0, output_keep_prob=drop_keep_rate)

    # 设置多层GRU
    stacked_fw_cell = tf.nn.rnn_cell.MultiRNNCell([gru_fw_cell] * layer_num, state_is_tuple=True)
    stacked_lw_cell = tf.nn.rnn_cell.MultiRNNCell([gru_lw_cell] * layer_num, state_is_tuple=True)

    # 初始化状态
    initial_fw_cell = stacked_fw_cell.zero_state(batch_size, tf.float32)
    initial_lw_cell = stacked_lw_cell.zero_state(batch_size, tf.float32)

    inputs = tf.unstack(inputs, axis=1)
    outputs, _, _ = tf.nn.static_bidirectional_rnn(stacked_fw_cell, stacked_lw_cell, inputs, initial_fw_cell,
                                                   initial_lw_cell, dtype=tf.float32)
    output = tf.stack(outputs, axis=1)
    output = tf.reshape(tf.concat(output, 1), [-1, embedding_size * 2])
    print(output.shape)

    # output layer
    softmax_weight = weight_variable([embedding_size * 2, pouncation_num])
    softmax_bias = bias_variable([pouncation_num])
    begin_pre_labels = tf.matmul(output, softmax_weight) + softmax_bias

    # 原始每段句子的标注结果标签概率
    begin_pre_labels_reshape = tf.reshape(begin_pre_labels, (-1, 512, 7))

    pre_labels_reshape = tf.nn.softmax(begin_pre_labels_reshape, axis=-1)

    print('------0', data_word.shape)
    print(pre_labels_reshape.shape)
    y_label_pre = tf.cast(tf.argmax(pre_labels_reshape, axis=-1), tf.int32)
    print('------1', y_label_pre.shape)

    y_label_reshape = tf.reshape(data_label, (-1, 512, 7))
    y_label_real_soft = tf.nn.softmax(y_label_reshape)

    y_label_real = tf.cast(tf.argmax(y_label_reshape, axis=-1), tf.int32)
    print('------2', y_label_real.shape)

    # prediction
    correct_prediction = tf.equal(y_label_pre, y_label_real)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    print('Prediction', correct_prediction, 'Accuracy', accuracy)
    print(data_label.shape)

    # loss
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(tf.cast(y_label_real_soft, dtype=tf.float32) * tf.log(pre_labels_reshape)))

    # model summary
    tf.summary.scalar('loss', cross_entropy)

    # train
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    grads = optimizer.compute_gradients(cross_entropy)
    for i, (g, v) in enumerate(grads):
        if g is not None:
            grads[i] = (tf.clip_by_norm(g, 5), v)  # 阈值这里设为5
    train_step_op = optimizer.apply_gradients(grads)
    # trian_step_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # Saver
    saver = tf.train.Saver(max_to_keep=10)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    gstep = 0
    Flag = 0

    # summarier
    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(join(summaries_dir, 'train'), sess.graph)

    best_score = 0
    num_epoch_no_improve = 0
    if isTrain:
        for epoch in range(epochs):
            print('当前轮次为：', epoch)
            tf.train.global_step(sess, global_step_tensor=global_step)
            sess.run(train_initial_op)
            for step in range(int(train_step)):
                smrs, loss, acc, gstep, _ = sess.run([summaries, cross_entropy, accuracy, global_step, train_step_op],
                                                     feed_dict={keep_prob: 1})

                if step % steps_per_print == 0:
                    print('Global Step', gstep, 'Step', step, 'Trian loss', loss, 'Train Accuracy', acc)

                    if gstep % steps_per_sumary == 0:
                        writer.add_summary(smrs, gstep)
                        print('Write Summaries to ', summaries_dir)

            if epoch % epochs_per_dev == 0:
                sess.run(dev_initial_op)
                for step in range(int(dev_step)):
                    # 此处使用的minibatch梯度下降,将数据集分成train_step份.在每一次小batch中更新参数
                    if step % steps_per_print == 0:
                        acc = sess.run(accuracy, feed_dict={keep_prob: 1})
                        if acc > best_score:
                            num_epoch_no_improve = 0
                            best_score = acc
                            saver.save(sess, save_dir + 'model.ckpt', global_step=gstep)
                            print('Dev Accuracy', acc, 'Step', step)
                        else:
                            num_epoch_no_improve += 1
                            learning_rate -= 0.05
                            if num_epoch_no_improve > num_epoch_no_improve_bear:
                                Flag = 1
                                break
            if Flag:
                break
    else:
        # load model
        last_ckpt = tf.train.latest_checkpoint(save_dir)
        if last_ckpt:
            saver.restore(sess, last_ckpt)
        print('Restore From', last_ckpt)
        sess.run(test_initial_op)

        for step in range(int(test_step)):
            data_word_result, data_label_pre_result, data_label_real_result, acc = sess.run(
                [data_word, begin_pre_labels_reshape, y_label_reshape, accuracy],
                feed_dict={keep_prob: 1})
            print("test step", step, '未处理前Accuracy', acc)
            # y_predict_label = np.reshape(data_label_result, data_word_result.shape)
            for i in range(len(data_word_result)):
                data_word_result, y_predict_label = list(filter(lambda x: x, data_word_result[i])), list(
                    filter(lambda x: x, data_label_pre_result[i]))
                word_x, label_y = ''.join(id2word[data_word_result].values), ''.join(
                    id2tag[y_predict_label].values)
                print(word_x, label_y)


# 未完成工作：将返回的汉字按照对应的标签序列进行分割,在B或者S的前面加空格，并打印出一段示例
if __name__ == '__main__':
    main()
