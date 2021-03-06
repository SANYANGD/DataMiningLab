import time
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report


# 读取数据返回x_train, y_train, x_test, y_test
def load_data(remainder):
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


# 按照John Platt的论文构造SMO的数据结构
class SVM_SMO:

    # 按照John Platt的论文构造SMO的数据结构
    def __init__(self, X, y, kernel):
        self.X = X  # 训练样本
        self.y = y  # 类别 label
        self.C = 5  # 正则化常量，调整（过）拟合的程度
        self.kernel = kernel  # 核函数，高斯（RBF）
        self.m, self.n = np.shape(self.X)  # 训练集大小（m）和特征数（n）
        self.alphas = np.zeros(self.m)  # 拉格朗日乘子，与样本一一相对
        self.b = 0.0  # B
        self.errors = np.zeros(self.m)  # E 用于存储alpha值实际与预测值得差值，与样本数量一一相对
        self.w = np.zeros(self.n)  # 初始化权重w的值，主要用于线性核函数
        self.tol = 0.01
        self.eps = 0.01  # error tolerance


# 高斯核函数 返回参数为sigma的数组x和y的高斯相似度
def gaussian_kernel(x, y, sigma=1):
    # 高斯核函数 返回参数为sigma的数组x和y的高斯相似度
    result = 0
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = np.exp(-(np.linalg.norm(x - y, 2)) ** 2 / (2 * sigma ** 2))
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
        result = np.exp(-(np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma ** 2))
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = np.exp(-(np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2) / (2 * sigma ** 2))
    return result


# 判别函数
def decision_function(model, test=-1, i=-1):
    if i is not -1:
        test_l = model.X[test]
    else:
        test_l = test
    # 将决策函数应用于'x_test'中的输入特征向量
    result = (model.alphas * model.y).dot(model.kernel(model.X, test_l)) - model.b

    return result


# 选择了alpha2, alpha1后开始第一步优化，然后迭代，“第二层循环，内循环”
# SMO主要的优化步骤在这里发生
def take_step(i1, i2, model):
    # alpha2, alpha1相等时跳出
    if i1 == i2:
        return 0, model
    # a1, a2 的原值，old value，优化在于产生优化后的值，新值 new value
    # 如下的alpha1, 2, y1, y2, E1, E2, s含义与论文一致
    alph1 = model.alphas[i1]
    alph2 = model.alphas[i2]

    y1 = model.y[i1]
    y2 = model.y[i2]

    E1 = get_error(model, i1)
    E2 = get_error(model, i2)
    s = y1 * y2

    L, H = 0, 0
    # 计算alpha的边界，L, H
    if y1 != y2:
        # y1,y2 异号，使用 Equation (J13)
        L = max(0, alph2 - alph1)
        H = min(model.C, model.C + alph2 - alph1)
    elif y1 == y2:
        # y1,y2 同号，使用 Equation (J14)
        L = max(0, alph1 + alph2 - model.C)
        H = min(model.C, alph1 + alph2)
    if L == H:
        return 0, model

    # 分别计算啊样本1, 2对应的核函数组合，目的在于计算eta 也就是求一阶导数后的值，目的在于计算a2new
    k11 = model.kernel(model.X[i1], model.X[i1])
    k12 = model.kernel(model.X[i1], model.X[i2])
    k22 = model.kernel(model.X[i2], model.X[i2])
    # 计算 eta，equation (J15)
    eta = k11 + k22 - 2 * k12

    # 如论文中所述，分两种情况根据eta为正positive 还是为负或0来计算计算a2 new
    if eta > 0:
        # equation (J16) 计算alpha2
        a2 = alph2 + y2 * (E1 - E2) / eta
        # 把a2夹到限定区间 equation （J17）
        if L < a2 < H:
            a2 = a2
        elif a2 <= L:
            a2 = L
        elif a2 >= H:
            a2 = H
    # 如果eta不为正（为负或0）
    else:
        # Equation （J19） 在特殊情况下，eta可能不为正
        f1 = y1 * (E1 + model.b) - alph1 * k11 - s * alph2 * k12
        f2 = y2 * (E2 + model.b) - s * alph1 * k12 - alph2 * k22

        L1 = alph1 + s * (alph2 - L)
        H1 = alph1 + s * (alph2 - H)

        Lobj = L1 * f1 + L * f2 + 0.5 * (L1 ** 2) * k11 + 0.5 * (L ** 2) * k22 + s * L * L1 * k12

        Hobj = H1 * f1 + H * f2 + 0.5 * (H1 ** 2) * k11 + 0.5 * (H ** 2) * k22 + s * H * H1 * k12

        if Lobj < Hobj - model.eps:
            a2 = L
        elif Lobj > Hobj + model.eps:
            a2 = H
        else:
            a2 = alph2

    # 当new a2千万分之一接近C或0是，就让它等于C或0
    if a2 < 1e-8:
        a2 = 0.0
    elif a2 > (model.C - 1e-8):
        a2 = model.C
    # 超过容差仍不能优化时，跳过
    if np.abs(a2 - alph2) < model.eps * (a2 + alph2 + model.eps):
        return 0, model

    # 根据新 a2计算 新 a1 Equation(J18)
    a1 = alph1 + s * (alph2 - a2)

    # 更新 bias b的值 Equation (J20)
    b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + model.b
    # equation (J21)
    b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + model.b

    # 根据a1、a2和L、H设置新的阈值
    if 0 < a1 < model.C:
        b_new = b1
    elif 0 < a2 < model.C:
        b_new = b2
    else:
        b_new = (b1 + b2) * 0.5
    model.b = b_new

    # 在alphas矩阵中分别更新a1, a2的值
    model.alphas[i1] = a1
    model.alphas[i2] = a2

    # 优化完了，更新差值矩阵的对应值，同时更新差值矩阵其它值
    model.errors[i1] = 0
    model.errors[i2] = 0
    # 更新差值 Equation (12)
    for i in range(model.m):
        if 0 < model.alphas[i] < model.C:
            model.errors[i] += y1 * (a1 - alph1) * model.kernel(model.X[i1], model.X[i]) + y2 * (
                    a2 - alph2) * model.kernel(model.X[i2], model.X[i]) + model.b - b_new

    return 1, model


# 计算E值
def get_error(model, i):
    if 0 < model.alphas[i] < model.C:
        return model.errors[i]
    else:
        return decision_function(model=model, i=i) - model.y[i]


# 确定alpha1, 也就是old alpha1，并送到take_step优化
def examine_example(i2, model):
    y2 = model.y[i2]
    alph2 = model.alphas[i2]
    E2 = get_error(model, i2)
    r2 = E2 * y2

    # 确定alpha1, 也就是old alpha1，并送到take_step优化
    # 下面条件之一满足，进入if开始找第二个alpha，送到take_step进行优化
    i1 = 0
    if (r2 < -model.tol and alph2 < model.C) or (r2 > model.tol and alph2 > 0):
        if len(model.alphas[(model.alphas != 0) & (model.alphas != model.C)]) > 1:
            # 选择Ei矩阵中差值最大的先优化
            # 要想|E1-E2|最大，只需要在E2为正时，选择最小的Ei作为E1，在E2为负时选择最大的Ei作为E1
            if model.errors[i2] > 0:
                i1 = np.argmin(model.errors)
            elif model.errors[i2] <= 0:
                i1 = np.argmax(model.errors)

            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model

        # 循环所有非0 非C alphas值进行优化，随机选择起始点
        for i1 in np.roll(np.where((model.alphas != 0) & (model.alphas != model.C))[0],
                          np.random.choice(np.arange(model.m))):
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model

        # a2确定的情况下，如何选择a1? 循环所有(m-1) alphas, 随机选择起始点
        for i1 in np.roll(np.arange(model.m), np.random.choice(np.arange(model.m))):
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model
    # 先看最上面的if语句，如果if条件不满足，说明KKT条件已满足，找其它样本进行优化，退出
    return 0, model


# 顺序执行所有样本(第一个if中的循环)，且else中的循环没有可优化的alpha，目标函数收敛了：
# 在容差之内，并且满足KKT条件则循环退出; 如果执行loop_num次仍未收敛，也退出。
# 重点在于确定old alpha2的下标，old alpha2和old alpha1都是启发式的选择
def routine(model):
    num_changed = 0
    exam_all = 1

    # 计数器，记录优化时的循环次数
    loop_num = 0
    loop_num1 = 0
    loop_num2 = 0

    # num_changed = 0, exam_all = 0时 循环退出
    # 顺序执行所有样本(第一个if中的循环)，且else中的循环没有可优化的alpha，目标函数收敛了：
    # 在容差之内，并且满足KKT条件则循环退出; 如果执行loop_num次仍未收敛，也退出。
    # 重点在于确定old alpha2的下标，old alpha2和old alpha1都是启发式的选择
    while (num_changed > 0) or exam_all:
        num_changed = 0
        if loop_num == 500:
            break
        loop_num += 1
        if exam_all:
            loop_num1 += 1  # 记录顺序选择alpha2时的循环次数
            # 顺序选择a2的，送给examine_example选择alpha1
            for i in range(model.alphas.shape[0]):
                examine_result, model = examine_example(i, model)
                num_changed += examine_result
        else:  # 上面if执行完后执行
            loop_num2 += 1
            # 遍历alpha尚未达到极限
            for i in np.where((model.alphas != 0) & (model.alphas != model.C))[0]:
                examine_result, model = examine_example(i, model)
                num_changed += examine_result
        if exam_all == 1:
            exam_all = 0
        elif num_changed == 0:
            exam_all = 1
    # print(loop_num, ',', loop_num1, ',', loop_num2)
    # return model


# 分类函数
def classification(model, test):
    i = 0
    x = []
    for a in model.alphas:
        if a >= 0:
            bs = model.y[i] - (model.alphas * model.y).dot(model.kernel(model.X, model.X[i]))
            x.append(bs)
        i += 1
    # 将决策函数应用于'x_test'中的输入特征向量
    return (model.alphas * model.y).dot(model.kernel(model.X, test)) + (sum(x) / len(x))  # model.b


def main(remainder):
    # 获取数据
    x_train, y_train, x_test, y_test = load_data(remainder)
    # x_train, y_train = make_blobs(n_samples=400, centers=2, n_features=12, random_state=2)
    # x_test, y_test = make_blobs(n_samples=20, centers=2, n_features=12, random_state=2)

    # 实例化模型
    svm_model = SVM_SMO(X=x_train, y=y_train, kernel=gaussian_kernel)

    # 初始化E
    initial_error = decision_function(model=svm_model, test=svm_model.X) - svm_model.y
    svm_model.errors = initial_error

    # 训练模型
    routine(svm_model)

    # 分类
    predict = classification(svm_model, test=x_test)
    predict[predict < 0] = -1
    predict[predict > 0] = 1

    # 打印结果
    print(classification_report(y_test, predict))

    return y_test, predict


if __name__ == '__main__':
    real = []
    predict = []
    for i in range(10):
        r, p = main(i)
        real.extend(r)
        predict.extend(p)
    print(classification_report(real, predict))
