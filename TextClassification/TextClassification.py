from collections import Counter
import pandas as pd
import jieba


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
        one = Counter(dict(filter(lambda x: 1 == x[1], c[i].items())))
        c[i] -= one
        c[i] = dict(c[i])

    return c


def conditional_probability(data):
    k = [[] for _ in range(10)]
    ks = []
    for i in range(0, 10):
        k[i] = list(data[i].keys())
        ks.extend(k[i])
    k_s = list(set(ks))
    # 得到训练集中所有的词（已去重）

    # 类别j文档集词列表，词语总数
    s = [0] * 10
    for j in range(0, 10):
        for dic in data[j]:
            s[j] += data[j][dic]

    # 类别m文档  条件概率
    p = [{} for _ in range(0, 10)]
    for m in range(0, len(data)):
        for n in range(0, len(k[m])):
            p[m].update({k[m][n]: (data[m][k[m][n]] + 1) / (len(k_s) + s[m])})
        for o in range(len(k[m]), len(k_s)):
            p[m].update({'q': 1 / (len(k_s) + s[m])})

    return p


def main(k):
    divide_train_test(k)
    x_train, y_train, x_test, y_test = data_load()
    train_data = jieba_cut(x_train)

    # 得到10种类别训练集的dict
    # count[ dict * 10 ]
    count = word_frequency(train_data)

    # 条件概率 con_pro[1-10][ 词 ]
    con_pro = conditional_probability(count)
    with open('model.txt', 'w+', encoding='UTF-8') as f:
        for i in range(len(con_pro)):
            for j in range(len(con_pro[i])):
                f.write(str(con_pro[i][j]))
                f.write(',')
            f.write('\n')
    print(con_pro[0])


if __name__ == '__main__':
    main(1)
