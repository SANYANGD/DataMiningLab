# ! python3
# -*- coding: utf-8 -*-
import csv
import jieba
from collections import Counter
import os


def getTxt(path):
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    txts = [[] for _ in range(100)]
    for i in range(0, 100):  # 遍历文件夹
        position = path + '//' + files[i]  # 构造绝对路径，"\\"，其中一个'\'为转义符
        print(position)
        with open(position, "r", encoding='utf-8') as f:  # 打开文件
            data = f.read()  # 读取文件
            txts[i].append(data)
    # txts = ','.join(txts)  # 转化为非数组类型
    # print(txts)
    return (txts)


def getWords(txt):
    # infile = open(file, 'r', encoding='UTF-8')
    # txt = infile.read()
    # infile.close()
    c = Counter()
    for i in range(0, len(txt)):
        txt_list = list(jieba.cut(','.join(txt[i])))
        for x in txt_list:
            if len(x) > 1 and x != '\r\n':
                c[x] += 1
    # txt_list = list(jieba.cut(txt))  # 结巴分词
    # c = Counter(txt_list)
    # print(dict(c))
    createDictCSV('out.csv', c)  # 将词频写入csv


def createDictCSV(filename, datadict):
    with open(filename, "w", encoding='UTF-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        for k, v in datadict.items():
            csv_writer.writerow([k, v])
        csvfile.close()


if __name__ == '__main__':
    t = getTxt('D://SanYang//program//DataMiningLab//TextClassification//news//constellation')
    getWords(t)
