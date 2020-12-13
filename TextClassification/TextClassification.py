import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

'''数据集划分，k取0-9'''


def divide_train_test(k):
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


'''结巴分词'''


def jieba_cut(data):
    words = data.apply(lambda x: ' '.join(jieba.cut(x)))
    return words


'''数据加载'''


def data_load():
    with open('cnews.train.txt', 'r', encoding='utf-8') as f_train:
        train = pd.read_table(f_train, names=['类别', '内容'])
    with open('cnews.test.txt', 'r', encoding='utf-8') as f_test:
        test = pd.read_table(f_test, names=['类别', '内容'])
    x_train = train['内容']
    y_train = train['类别']
    x_test = test['内容']
    y_test = test['类别']
    return x_train, y_train, x_test, y_test


'''停词过滤'''


def get_stopword():
    with open('cnews.vocab.txt', 'r', encoding='utf-8') as sw:
        sw_list = sw.readlines()
        sws = [x.strip() for x in sw_list]
    return sws

def main():
    divide_train_test(1)
    x_train, y_train, x_test, y_test = data_load()

    train_data = jieba_cut(x_train)

    # stopwords = get_stopword()
    # tv = TfidfVectorizer(stop_words=stopwords, max_features=5000, lowercase=False)
    # tv.fit(train_data)
    #
    # # 贝叶斯分类
    # model = MultinomialNB(alpha=0.2)  # 参数自己选,当然也可以不特殊设置
    # model.fit(tv.transform(jieba_cut(x_train)), y_train)  # 训练
    # # 结果0.91
    # print(model.score(tv.transform(jieba_cut(x_test)), y_test))  # 测试
    # # 打印概率
    # # model.predict_proba(tv.transform(jieba_cut(x_test)))
    # print('其他指标:\n', classification_report(y_test, model.predict(tv.transform(jieba_cut(x_test)))))

    # 不用tfidf
    vec = CountVectorizer()
    x_train = vec.fit_transform(jieba_cut(x_train))
    x_test = vec.transform(jieba_cut(x_test))
    model = MultinomialNB(alpha=0.2)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    print('其他指标:\n', classification_report(y_test, y_predict))


if __name__ == '__main__':
    main()