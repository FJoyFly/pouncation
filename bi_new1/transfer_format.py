import codecs
import os
import os.path
import re
import numpy as np

file_path = '/home/joyfly/桌面/全部/别史'
outfile = '/home/joyfly/桌面/宋书'
in_output_file = '/home/joyfly/桌面/宋2'
output_file = '/home/joyfly/桌面/副本2'

length_526 = 0
all_length = 0
length = 0
num_none = 0


def search_txt(file_path, outfile):
    global length, all_length, num_none  # 用来控制文本中没有有标点符号的段落数
    filenames = os.listdir(file_path)
    f = codecs.open('/home/joyfly/桌面/宋书', 'w', 'utf-8')
    new_filenames = filenames
    for filename in new_filenames:
        newdir = os.path.join(file_path, filename)
        print('Print path', newdir)
        new_filenames = newdir
        if os.path.isfile(newdir):
            if os.path.splitext(newdir)[1] == '.txt':
                for line in open(newdir):
                    if line:
                        if len(line) > 512:
                            length += 1
                        if len(line) < 256:
                            if len(line) > 32:
                                word = re.findall('[，。:“”？！；]', line)
                                if word is None:
                                    num_none += 1
                                    if num_none > 5:
                                        break
                                    continue
                                num_none = 0
                                all_length += 1
                                f.write(line)
                            else:
                                all_length += 1
                                num_none = 0
                                f.write(line)
                f.write('\n')
        elif os.path.isdir(newdir):
            search_txt(new_filenames, outfile)
    f.close()


# 将初始文本格式转换为可以处理的格式，在符号前后加空格
def change_format(input_file, output_file):
    """
    :param input_file: 输入初始文本
    :param output_file:  输出初整理文本
    :return:
    """
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for new_data in input_data.readlines():
        if new_data == '\n':
            continue
        else:
            for i in new_data:
                # 如果是所需要标点中的一个，使用的时UNICODE编码
                if i == u'\u3002' or i == u'\uFF1F' or i == u'\uFF01' \
                        or i == u'\uFF0C' or i == u'\uFF1B' or i == u'\uFF1A' \
                        or i == u'\u201C' or i == u'\u201D' \
                        or i == u'\u3001':
                    output_data.write(' ' + i + ' ')
                elif u'\u4E00' <= i <= u'\u9FEF':
                    output_data.write(i)
                else:
                    continue
            output_data.write("\n")
    input_data.close()
    output_data.close()


def score_transfer(inputfile):
    num_of_length = np.zeros(6)
    data = codecs.open(inputfile, 'r', 'utf-8')
    for i in data.readlines():
        list = re.split("[，。？：；‘’“”、]", i)
        for _ in list:
            _ = _.strip()
            print(_, len(_))
            number_length = len(_)
            if number_length > 5:
                num_of_length[5] += 1
            elif number_length == 5:
                num_of_length[4] += 1
            elif number_length == 4:
                num_of_length[3] += 1
            elif number_length == 3:
                num_of_length[2] += 1
            elif number_length == 2:
                num_of_length[1] += 1
            elif number_length == 1:
                num_of_length[0] += 1
    return num_of_length


# 将文本后标注上对应的标签
def character_tagging(input_file, output_file):
    '''

    :param input_file:初整理文本
    :param output_file: 终整理文本
    :return: 已标注文本
    '''
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        word_list = line.strip().split()  # 这里读取数据时按\n进行的
        for word in word_list:
            if u'\u4E00' <= word <= u'\u9FEF':
                if len(word) == 1:
                    output_data.write(word + "/S" + '  ')
                else:
                    output_data.write(word[0] + "/B" + '  ')
                    for w in word[1:len(word) - 3]:
                        output_data.write(w + "/M" + '  ')
                if len(word) > 3:
                    output_data.write(
                        word[len(word) - 3] + "/E3" + '  ' + word[len(word) - 2] + "/E2" + '  ' + word[
                            len(word) - 1] + "/E" + '  ')
                elif len(word) > 2:
                    output_data.write(word[len(word) - 2] + "/E2" + '  ' + word[len(word) - 1] + "/E" + '  ')
                elif len(word) > 1:
                    output_data.write(word[len(word) - 1] + "/E" + '  ')
        output_data.write("\n")
    input_data.close()
    output_data.close()


# 计算次文本中总共不重复的字数
def count_vocab(input_file):
    '''

    :param input_file: 初整理文本
    :return:
    '''
    input_data = codecs.open(input_file, 'r', 'utf-8')
    data_vocabu = []
    for data_vo in input_data.readlines():
        for i in data_vo:
            if u'\u4E00' <= i <= u'\u9FEF' and i not in data_vocabu:
                data_vocabu.extend(i)
            else:
                continue
    return data_vocabu


print('正在查找当前目录下带标点txt文件\n')
search_txt(file_path, outfile)
print('正在将初始文件转换成所需格式文件\n')
change_format(outfile, in_output_file)
print('正在将格式文件中古文标注对应的标签')
# count_vocab(in_output_file)
character_tagging(in_output_file, output_file)
print('段落总数：', all_length)
print('超过固定长度的段落数：', length)
num_of_lenght = score_transfer(in_output_file)
print('各句子长度数:', num_of_lenght)
