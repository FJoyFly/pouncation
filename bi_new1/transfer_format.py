import codecs
import os
import os.path
import re

file_path = '/home/joyfly/桌面/全部/别史'
outfile = '/home/joyfly/桌面/宋书'
in_output_file = '/home/joyfly/桌面/宋2'
output_file = '/home/joyfly/桌面/副本2'

# length_526 = 0
all_length = 0
length = 0
num_word_all = 0
line_point = 0


# line_most = 300000


# 用来对文本进行迭代取其中内容
def search_txt(file_path, outfile):
    global line_point  # 控制写入文本的段落数
    # if line_point > line_most:
    compile = '[，。！？：；“”]'
    filenames = os.listdir(file_path)
    f = codecs.open(outfile, 'a+', 'utf-8')
    new_filenames = filenames
    for filename in new_filenames:
        newdir = os.path.join(file_path, filename)
        print('Print path', newdir)
        new_filenames = newdir
        if os.path.isfile(newdir):
            if os.path.splitext(newdir)[1] == '.txt':
                for line in open(newdir):
                    if line:
                        if len(line) > 32:  # 对原始文段清洗一遍,判断是否是带有标点的段落.是,则加入
                            word = re.findall(compile, line)
                            if word is None:
                                break
                            if 256 < len(line) <= 512:
                                middle = len(line) // 2
                                for i in range(middle):
                                    if re.match(compile, line[middle + i]):
                                        f.write(line[:middle + i] + '\n')
                                        line_point += 1
                                        f.write(line[middle + i + 1:])
                                        line_point += 1
                            elif len(line) < 256:
                                f.write(line)
                                line_point += 1
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
    global num_word_all
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for new_data in input_data.readlines():
        if new_data == '\n':
            continue
        else:
            for i in new_data:
                # 如果是所需要标点中的一个，使用的时UNICODE编码
                t = re.search('[，。：；？！、]', i)
                if t:
                    # if i == u'\u3002' or i == u'\uFF1F' or i == u'\uFF01' \
                    #         or i == u'\uFF0C' or i == u'\uFF1B' or i == u'\uFF1A' \
                    #         or i == u'\u201C' or i == u'\u201D' \
                    #         or i == u'\u3001':
                    output_data.write(' ' + i + ' ')
                elif u'\u4E00' <= i <= u'\u9FEF':
                    num_word_all += 1
                    output_data.write(i)
                else:
                    continue
            output_data.write("\n")
    input_data.close()
    output_data.close()


# # 大概计算各标签间转移概率
# def score_transfer(inputfile):
#     num_of_length = np.zeros(6)
#     data = codecs.open(inputfile, 'r', 'utf-8')
#     for i in data.readlines():
#         list = re.split("[，。？：；‘’“”、]", i)
#         for _ in list:
#             _ = _.strip()
#             number_length = len(_)
#             if number_length > 5:
#                 num_of_length[5] += 1
#             elif number_length == 5:
#                 num_of_length[4] += 1
#             elif number_length == 4:
#                 num_of_length[3] += 1
#             elif number_length == 3:
#                 num_of_length[2] += 1
#             elif number_length == 2:
#                 num_of_length[1] += 1
#             elif number_length == 1:
#                 num_of_length[0] += 1
#     return num_of_length


# 将文本后标注上对应的标签
def character_tagging(input_file, output_file):
    '''

    :param input_file:初整理文本
    :param output_file: 终整理文本
    :return: 已标注文本
    '''
    global all_length, tag
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        if line == '\n':
            continue
        all_length += 1
        word_list = line.strip().split()  # 这里读取数据时按\n进行的
        for i, word in enumerate(word_list):
            if u'\u4E00' <= word <= u'\u9FEF':
                if line[i + 1] == '，':
                    tag = 'd'
                elif line[i + 1] == '。':
                    tag = 'j'
                elif line[i + 1] == '？':
                    tag = 'w'
                elif line[i + 1] == '！':
                    tag = 'g'
                elif line[i + 1] == '；':
                    tag = 'f'
                elif line[i + 1] == '：':
                    tag = 'm'
                elif line[i + 1] == '、':
                    tag = 't'
                else:
                    continue
                if len(word) == 1:
                    output_data.write(word + '/' + tag + '  ')
                else:
                    output_data.write(word[0] + "/B" + '  ')
                    for w in word[1:len(word) - 3]:
                        output_data.write(w + "/M" + '  ')
                if len(word) > 3:
                    output_data.write(
                        word[len(word) - 3] + "/F" + '  ' + word[len(word) - 2] + "/G" + '  ' + word[
                            len(word) - 1] + '/' + tag + '  ')
                elif len(word) > 2:
                    output_data.write(word[len(word) - 2] + "/G" + '  ' + word[len(word) - 1] + '/' + tag + '  ')
                elif len(word) > 1:
                    output_data.write(word[len(word) - 1] + '/' + tag + '  ')
        output_data.write("\n")
    input_data.close()
    output_data.close()


# # 计算次文本中总共不重复的字数
# def count_vocab(input_file):
#     '''
#
#     :param input_file: 初整理文本
#     :return:
#     '''
#     input_data = codecs.open(input_file, 'r', 'utf-8')
#     data_vocabu = []
#     for data_vo in input_data.readlines():
#         for i in data_vo:
#             if u'\u4E00' <= i <= u'\u9FEF' and i not in data_vocabu:
#                 data_vocabu.extend(i)
#             else:
#                 continue
#     input_data.close()
#     return data_vocabu


print('正在查找当前目录下带标点txt文件\n')
search_txt(file_path, outfile)
print('正在将初始文件转换成所需格式文件\n')
change_format(outfile, in_output_file)
print('正在将格式文件中古文标注对应的标签\n')
character_tagging(in_output_file, output_file)
print('段落总数：', all_length)
print('超过固定长度的段落数：', length)
# print(len(count_vocab(in_output_file)))
# print('各句子长度数:', num_of_lenght)
print(num_word_all)
