import numpy as np


class SVM:
    def __init__(self, data, tag, maxiter, C):
        self.X = data
        self.Y = tag
        self.m = len(tag)  # 样本数
        self.alphas = [0] * self.m  # 阿尔法参数数组
        self.b = 0
        self.maxiter = maxiter  # 最大迭代次数
        self.C = C  # 惩罚因子
        self.kernelValue = RBF(data)  # 核函数矩阵
        self.fx = [0] * self.m
        self.E = [0] * self.m


def loaddata(remainder):
    filepath = 'dataset.csv'
    data = np.loadtxt(filepath, dtype=np.float, delimiter=',', usecols=range(12), encoding='utf-8')
    data_tag = np.loadtxt(filepath, dtype=np.int, delimiter=',', usecols=12, encoding='utf-8')
    x_train, y_train, x_test, y_test = [], [], [], []
    for i in range(len(data_tag)):
        if i % 10 == remainder:  # 按余数划分数据集
            x_test.append(data[i])
            y_test.append(data_tag[i])
        else:
            x_train.append(data[i])
            y_train.append(data_tag[i])
    # 返回numpy数组
    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test)


# 高斯核的实现方法
def RBF(x, L, sigma):
    '''
    x: 待分类的点的坐标
    L: 某些中心点，通过计算x到这些L的距离的和来判断类别
    '''
    return np.exp(-(np.sum((x - L) ** 2)) / (2 * sigma ** 2))


def g(svm):


def alpha_select(svm):



def main(remainder):
    x_train, y_train, x_test, y_test = loaddata(remainder)
    svm = SVM(x_train, y_train, maxiter=999, C=1)
    # alpha_select(alpha, 1)


if __name__ == '__main__':
    main(0)
