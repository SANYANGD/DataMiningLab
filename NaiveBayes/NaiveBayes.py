# -*-coding: utf-8 -*-

import csv
import math
import numpy as np
import re
import pandas as pd
from sklearn.model_selection import train_test_split


# 去除文件中@开头的行
def writeFile(filename):
    with open(filename, mode='r') as f:
        data = f.read()
        result = data.split('\n')
        with open('./titanic.csv', mode='w') as file:
            file.writelines('Class,Age,Sex,Survived' + '\n')
            for i in result:
                if not re.match(r'@.*', i):
                    file.writelines(i + '\n')


# UTF-8编码格式csv文件数据读取
def csv2arr(filename):
    df = pd.read_csv(filename)  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
    df.columns = ['Class', 'Age', 'Sex', 'Survived']
    data = df[['Class', 'Age', 'Sex', 'Survived']]
    data = np.array(data)
    return data


# 计算香农熵 pklog2pk
def countH(m, n):
    return -m / (m + n) * math.log(m / (m + n), 2) - n / (m + n) * math.log(n / (m + n), 2)


# 统计属性中的元素数量
def countP(data):
    data_cnt = {}  # 将结果用一个字典存储
    # 统计结果
    for d in data:
        # get(value, num)函数的作用是获取字典中value对应的键值, num=0指示初始值大小。
        data_cnt[d] = data_cnt.get(d, 0) + 1
    # 打印输出结果
    return data_cnt


# 将连续数据离散化
def attLisan(data):
    tempI = [0, 0, 0]
    for j in range(0, 3):
        att = countP(data[:, j])
        attKey = sorted(att.keys())
        temp = 0
        m = 0
        n = sum(att.values())
        for i in attKey:
            m = m + att[i]
            n = n - att[i]
            if m == 0 or n == 0: break
            if countH(m, n) >= temp:
                temp = countH(m, n)
                tempI[j] = i
    return tempI


# 将连续数据离散化
def attLisan2(data, th):
    for i in range(0, 3):
        for j in range(0, len(data)):
            if data[j, i] > th[i]: data[j, i] = 1
            else: data[j, i] = 0


# # 将连续数据离散化
# def attLisanOld(data, th):
#     for i in range(0, len(data)):
#         for j in range(0, 3):
#             if j == 0 and data[i, j] < -0.4275:
#                 data[i, j] = 0
#             elif j == 0 and data[i, j] >= -0.4275:
#                 data[i, j] = 1
#             elif j == 1 and data[i, j] < 2.076:
#                 data[i, j] = 0
#             elif j == 1 and data[i, j] >= 2.076:
#                 data[i, j] = 1
#             elif j == 2 and data[i, j] < -0.6995:
#                 data[i, j] = 0
#             elif j == 2 and data[i, j] >= -0.6995:
#                 data[i, j] = 1


# 数据集划分
def dataSplit(data):
    a = data[:, 0:3]
    t = data[:, 3]
    train_a, test_a, train_t, test_t = \
        train_test_split(a, t, test_size=0.3)
    # train_d训练集0.7, test_d测试集0.3, train_t训练集标签0.7, test_t测试集标签0.3
    return train_a, test_a, train_t, test_t


# 计算先验概率
# p_at[Class1_Survived1, Class0_Survived1,
#      Age1_Survived1,   Age0_Survived1,
#      Sex1_Survived1,   Sex0_Survived1,
#      Class1_Survived0, Class0_Survived0,
#      Age1_Survived0,   Age0_Survived0,
#      Sex1_Survived0,   Sex0_Survived0,]
# p_a[Class1, Class0,
#     Age1,   Age0,
#     Sex1,   Sex0_]
# p_t[Survived1, Survived0]
def naiveBayes(train_a, train_t):
    count_t = [0, 0]
    count_a = [0, 0, 0, 0, 0, 0]
    count_at = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in train_t:
        if i == 1:
            count_t[0] = count_t[0] + 1
        else:
            count_t[1] = count_t[1] + 1

    for i in range(0, len(train_t)):
        if train_a[i, 0] == 1:
            count_a[0] = count_a[0] + 1
        elif train_a[i, 0] == 0:
            count_a[1] = count_a[1] + 1
        if train_a[i, 1] == 1:
            count_a[2] = count_a[2] + 1
        elif train_a[i, 1] == 0:
            count_a[3] = count_a[3] + 1
        if train_a[i, 2] == 1:
            count_a[4] = count_a[4] + 1
        elif train_a[i, 2] == 0:
            count_a[5] = count_a[5] + 1

        if train_t[i] == 1 and train_a[i, 0] == 1:
            count_at[0] = count_at[0] + 1
        elif train_t[i] == 1 and train_a[i, 0] == 0:
            count_at[1] = count_at[1] + 1
    # for i in range(0, len(train_t)):
        if train_t[i] == 1 and train_a[i, 1] == 1:
            count_at[2] = count_at[2] + 1
        elif train_t[i] == 1 and train_a[i, 1] == 0:
            count_at[3] = count_at[3] + 1
    # for i in range(0, len(train_t)):
        if train_t[i] == 1 and train_a[i, 2] == 1:
            count_at[4] = count_at[4] + 1
        elif train_t[i] == 1 and train_a[i, 2] == 0:
            count_at[5] = count_at[5] + 1
    # for i in range(0, len(train_t)):
        if train_t[i] == -1 and train_a[i, 0] == 1:
            count_at[6] = count_at[6] + 1
        elif train_t[i] == -1 and train_a[i, 0] == 0:
            count_at[7] = count_at[7] + 1
    # for i in range(0, len(train_t)):
        if train_t[i] == -1 and train_a[i, 1] == 1:
            count_at[8] = count_at[8] + 1
        elif train_t[i] == -1 and train_a[i, 1] == 0:
            count_at[9] = count_at[9] + 1
    # for i in range(0, len(train_t)):
        if train_t[i] == -1 and train_a[i, 2] == 1:
            count_at[10] = count_at[10] + 1
        elif train_t[i] == -1 and train_a[i, 2] == 0:
            count_at[11] = count_at[11] + 1

    p_t = [count_t[0] / len(train_t), count_t[1] / len(train_t)]
    p_a = [count_a[0] / len(train_t), count_a[1] / len(train_t),
           count_a[2] / len(train_t), count_a[3] / len(train_t),
           count_a[4] / len(train_t), count_a[5] / len(train_t)]
    p_at = [count_at[0] / count_t[0], count_at[1] / count_t[0],
           count_at[2] / count_t[0], count_at[3] / count_t[0],
           count_at[4] / count_t[0], count_at[5] / count_t[0],
           count_at[6] / count_t[1], count_at[7] / count_t[1],
           count_at[8] / count_t[1], count_at[9] / count_t[1],
           count_at[10] / count_t[1], count_at[11] / count_t[1]]
    return p_at, p_t, p_a


# 后验概率计算
def classifier(test_a, test_t, p_at, p_t, p_a):
    r = 0
    for i in range(0, len(test_a)):
        yesTemp, noTemp, ptemp = 1, 1, 1
        if test_a[i, 0] == 1:
            yesTemp = yesTemp * p_at[0]
            noTemp = noTemp * p_at[6]
            ptemp = ptemp * p_a[0]
        elif test_a[i, 0] == 0:
            yesTemp = yesTemp * p_at[1]
            noTemp = noTemp * p_at[7]
            ptemp = ptemp * p_a[1]
        if test_a[i, 1] == 1:
            yesTemp = yesTemp * p_at[2]
            noTemp = noTemp * p_at[8]
            ptemp = ptemp * p_a[2]
        elif test_a[i, 1] == 0:
            yesTemp = yesTemp * p_at[3]
            noTemp = noTemp * p_at[9]
            ptemp = ptemp * p_a[3]
        if test_a[i, 2] == 1:
            yesTemp = yesTemp * p_at[4]
            noTemp = noTemp * p_at[10]
            ptemp = ptemp * p_a[4]
        elif test_a[i, 2] == 0:
            yesTemp = yesTemp * p_at[5]
            noTemp = noTemp * p_at[11]
            ptemp = ptemp * p_a[5]
        yesTemp = yesTemp * p_t[0] / ptemp
        noTemp = noTemp * p_t[1] / ptemp
        if yesTemp > noTemp:
            temp = 1
        else:
            temp = -1
        if temp == test_t[i]:
            r = r + 1
        print("类别%s的标签为1的后验概率为：%f，标签为-1的后验概率为：%f，预测结果其类别为：%d，实际类别为：%d。"
              % (test_a[i, :], yesTemp, noTemp, temp, test_t[i]))
    return r / len(test_t)


# 主函数
def main():
    # writeFile('./titanic.dat')
    dataset = csv2arr('./titanic.csv')
    threshold = attLisan(dataset)
    attLisan2(dataset, threshold)
    train_att, test_att, train_tag, test_tag = dataSplit(dataset)
    p_att_tag, p_tag, p_att = naiveBayes(train_att, train_tag)
    right = classifier(test_att, test_tag, p_att_tag, p_tag, p_att)
    print('正确率为：%f' % (right))


if __name__ == '__main__':
    main()
