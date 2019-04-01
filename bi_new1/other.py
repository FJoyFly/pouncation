import re
import numpy as np
import pandas as pd

# /home/joyfly/下载/1372394625/msr_train.txt
s = open('/home/joyfly/桌面/副本').read()
s = s.split('\r\n')
s = u''.join(s)
s = re.split(u'[，。！？、：；“”]/[S]', s)
data = []
label = []


def get_xy(s):
    s = re.findall('(.)/([BMES].)', s)
    if s:
        s = np.array(s)
        return list(s[:, 0]), list(s[:, 1])


for _ in s:
    x = get_xy(_)
    if x:
        data.append(x[0])
        label.append(x[1])

maxlen = 32

d = pd.DataFrame(index=range(len(data)))
d['data'] = data
d['label'] = label
d = d[d['data'].apply(len) <= maxlen]
d.index = range(len(d))
tag = pd.Series({'S ': 0, 'B ': 1, 'M ': 2, 'E3': 3, 'E2': 4, 'E ': 5, 'X ': 6})
chars = []
for i in data:
    chars.extend(i)

chars = pd.Series(chars).value_counts()
chars[:] = range(1, len(chars) + 1)

from keras.utils import np_utils

d['x'] = d['data'].apply(lambda x: np.array(list(chars[x]) + [0] * (maxlen - len(x))))


def trans_one(x):
    _ = map(lambda y: np_utils.to_categorical(y, 7), tag[x].values.reshape((-1, 1)))
    _ = list(_)
    _.extend([np.array([[0, 0, 0, 0, 0, 0, 1]])] * (maxlen - len(x)))
    return np.array(_)


d['y'] = d['label'].apply(trans_one)

embedding_size = 256
maxlen = 32
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
from keras.models import Model

sequence = Input(shape=(maxlen,), dtype='int32')
embedded = Embedding(len(chars) + 1, embedding_size, input_length=maxlen, mask_zero=True)(sequence)
bilstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
output = TimeDistributed(Dense(7, activation='softmax'))(bilstm)
model = Model(inputs=sequence, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
batch_size = 256
history = model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1, maxlen, 7)),
                    batch_size=batch_size,
                    epochs=50)

zy = {
    'S B ': 0.5,
    'S S ': 0.5,
    'B M ': 0.5,
    'B E3': 0.5,
    'B E2': 0.5,
    'B E ': 0.5,
    'M M ': 0.5,
    'M E3': 0.5,
    'E3E2': 1,
    'E2E ': 1,
    'E S ': 0.5,
    'E B ': 0.5
}

zy = {i: np.log(zy[i]) for i in zy.keys()}


def viterbi(nodes):
    paths = {'B ': nodes[0]['B '], 'S ': nodes[0]['S ']}
    for l in range(1, len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                if j[-1] + i in zy.keys():
                    nows[j + i] = paths_[j] + nodes[l][i] + 1.4 * zy[j[-1] + i]
            k = np.argmax(list(nows.values()))
            paths[list(nows.keys())[k]] = list(nows.values())[k]
    return list(paths.keys())[np.argmax(list(paths.values()))]


def simple_cut(s):
    if s:
        r = model.predict(np.array([list(chars[list(s)].fillna(0).astype(int)) + [0] * (maxlen - len(s))]),
                          verbose=False)[0][:len(s)]
        r = (-1) * np.log(r)
        print(r.shape)
        nodes = [dict(zip(['S ', 'B ', 'M ', 'E3', 'E2', 'E '], i[:6])) for i in r]
        print(nodes)
        t = viterbi(nodes)
        words = []
        for i in range(len(s)):
            if t[i] in ['S ', 'B ']:
                words.append(s[i])
            else:
                words.extend(s[i])
        return words
    else:
        return []


not_cuts = re.compile(u'([\da-zA-Z ]+)|[。，、？！\.\?,!]')


def cut_word(s):
    result = []
    j = 0
    for i in not_cuts.finditer(s):
        result.extend(simple_cut(s[j:i.start()]))
        result.append(s[i.start():i.end()])
        j = i.end()
    result.extend((simple_cut(s[j:])))
    return result


test = '永和九年'
print(cut_word(test))
