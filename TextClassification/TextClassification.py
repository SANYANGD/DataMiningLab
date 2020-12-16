from collections import Counter
import pandas as pd
import jieba
import json
import math
from sklearn.metrics import classification_report


def divide_train_test(k):
    # 数据集划分，k取0-9
    i = 0
    ftest = open('cnews.test.txt', 'w+', encoding='UTF-8')
    ftrain = open('cnews.train.txt', 'w+', encoding='UTF-8')
    with open('cnews.txt', 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            i = i + 1
            if i % 10 == k:
                ftest.writelines(line)
            else:
                ftrain.writelines(line)
    ftrain.close()
    ftest.close()


def jieba_cut(data):
    # 结巴分词
    txt_list = [[] for _ in range(10)]
    stopwords = get_stopword()
    j = 0
    for i in range(0, len(data)):
        txt_s = jieba.cut(data[i])
        for word in txt_s:
            if word not in stopwords and word != ' ' and word != '\xa0':
                txt_list[j].append(word)
        if (i + 1) % 90 == 0:
            j += 1
    # [[分词结果]*10]
    return txt_list


def data_load():
    # 数据加载
    with open('cnews.train.txt', 'r', encoding='utf-8') as f_train:
        train = pd.read_table(f_train, names=['类别', '内容'])
    with open('cnews.test.txt', 'r', encoding='utf-8') as f_test:
        test = pd.read_table(f_test, names=['类别', '内容'])
    x_train = train['内容']
    y_train = train['类别']
    x_test = test['内容']
    y_test = test['类别']
    return x_train, y_train, x_test, y_test


def get_stopword():
    # 停词
    with open('cnews.vocab.txt', 'r', encoding='utf-8') as sw:
        sw_list = sw.readlines()
        sws = [x.strip() for x in sw_list]
    return sws


def word_frequency(data):
    c = [[] for _ in range(10)]
    for i in range(0, len(data)):
        c[i] = Counter(data[i])

        # 去除值为1的元素
        d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        delete = Counter(dict(filter(lambda x: x[1] in d, c[i].items())))
        c[i] -= delete
        c[i] = dict(c[i])

    return c


def conditional_probability(data):
    k = [[] for _ in range(10)]
    ks = []
    for i in range(0, 10):
        k[i] = list(data[i].keys())
        ks.extend(k[i])
    k_s = list(set(ks))
    # for i in k_s:
    #     if not i in k_s:  # 去重
    #         k_s.append(i)
    # 得到训练集中所有的词（已去重）

    # 类别j文档集词列表，词语总数
    s = [0] * 10
    for j in range(0, 10):
        for dic in data[j]:
            s[j] += data[j][dic]

    # 类别m文档  条件概率
    p = [{} for _ in range(0, 10)]
    for m in range(0, len(data)):
        for n in range(0, len(k_s)):
            if k_s[n] not in data[m]:
                p[m].update({k_s[n]: math.log(1 / (len(k_s) + s[m]))})
            else:
                p[m].update({k_s[n]: math.log((data[m][k_s[n]] + 1) / (len(k_s) + s[m]))})

    return p


def test_data(c_p, t):
    pp = [{} for _ in range(len(t))]
    xt = [[] for _ in range(len(t))]
    tag = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
    t_tag = []
    # 100个测试文本
    for i in range(len(t)):
        xt[i] = list(jieba.cut(t[i]))
        # 某一测试文本对应10个类的后验概率
        for j in range(10):
            p = 1
            # for k in xt[i]:
            # 优化正确率，dict(Counter(xt[i]).most_common(30)).keys()表示只计算测试文本中最常见的30个词
            for k in dict(Counter(xt[i]).most_common(50)).keys():
                if k in c_p[j]:
                    for number in range(Counter(xt[i])[k]):
                        p *= c_p[j][k]
            pp[i].update({tag[j]: p})
        t_tag.append(max(pp[i], key=pp[i].get))
    return t_tag


def evaluate(predict, real):
    #                                                  predict
    #                 '体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经'
    #           '体育',
    #           '娱乐',
    #           '家居',
    #           '房产',
    #           '教育',
    #   real    '时尚',
    #           '时政',
    #           '游戏',
    #           '科技',
    #           '财经',
    e = [[0 for col in range(10)] for row in range(10)]
    n = -1
    for m in range(len(predict)):
        if m % 10 == 0: n += 1
        if predict[m] == '体育': e[n][0] += 1
        if predict[m] == '娱乐': e[n][1] += 1
        if predict[m] == '家居': e[n][2] += 1
        if predict[m] == '房产': e[n][3] += 1
        if predict[m] == '教育': e[n][4] += 1
        if predict[m] == '时尚': e[n][5] += 1
        if predict[m] == '时政': e[n][6] += 1
        if predict[m] == '游戏': e[n][7] += 1
        if predict[m] == '科技': e[n][8] += 1
        if predict[m] == '财经': e[n][9] += 1

    accuracy = [0 for col in range(10)]
    precision = [0 for col in range(10)]
    recall = [0 for col in range(10)]
    f1 = [0 for col in range(10)]
    for i in range(10):
        accuracy[i] = 0
        precision[i] = 0
        recall[i] = 0


def main(k):
    divide_train_test(k)
    x_train, y_train, x_test, y_test = data_load()
    tag = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']

    train_data = jieba_cut(x_train)

    # 得到10种类别训练集的dict , count[ dict * 10 ]
    count = word_frequency(train_data)

    # 条件概率 con_pro[1-10]{词：条件概率}
    con_pro = conditional_probability(count)

    # 输出测试的tag
    test_tag = test_data(con_pro, x_test)

    ri = classification_report(y_test, test_tag)
    print(ri)

    # evaluate(test_tag, y_test)

    # for i in range(len(test_tag)):
    #     print(test_tag[i], end=' ')
    #     if (i + 1) % 10 == 0:
    #         print(' ')

    return ri, y_test, test_tag


if __name__ == '__main__':
    # True Positive (TP): 把正样本成功预测为正。
    # True Negative (TN)：把负样本成功预测为负。
    # False Positive (FP)：把负样本错误地预测为正。
    # False Negative (FN)：把正样本错误的预测为负。
    result = [[] for row in range(10)]
    real = []
    predict = []
    for k in range(10):
        result[k], r, p = main(k)
        real.extend(r)
        predict.extend(p)
    a = classification_report(real, predict)
    print(a)
