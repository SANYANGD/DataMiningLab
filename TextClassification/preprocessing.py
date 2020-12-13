
import pandas as pd
import jieba


txt = pd.read_csv('cnews.txt', sep='\t', names=['label', 'content'])


def read_category(y_train):
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    categories = [x for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))
    label_id = []
    for i in range(len(y_train)):
        label_id.append(cat_to_id[y_train[i]])
    return label_id


txt_target = txt['label']
txt_label = read_category(txt_target)


def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))


# 添加jieba分词
txt_content = txt['content'].apply(chinese_word_cut)

print(txt_label)
