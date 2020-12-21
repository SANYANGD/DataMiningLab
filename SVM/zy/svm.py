import math
import fileRead
import numpy as np


# 定义一个模型的诸多变量
# 对于我们需要进行
class Model:
    def __init__(self, trainData, type="rbf"):
        self.trainData = trainData
        self.trainCount = len(trainData)
        self.aerfa = [0] * len(trainData)
        self.EList = [0] * len(trainData)
        self.yuzhiB = 0
        self.type = type
        self.Cpara = 50
        self.maxEpcho = 100
        self.e = 0.0001


def heFunction(x, z, model):
    """
    核函数方程
    :param x: 向量x
    :param z: 向量在
    :param model: 正在进行训练的model，根据model
    :return: 一个数值
    """
    # 创建指定长度的list的方法
    lamda = 0.5  # lamda大于0，需要调整参数设定他的值  默认的参数是类别的个数的倒数
    result = [0] * len(x)
    finalResult = 0
    # 高斯函数，又称为径向基函数（RBF），是非线性分类SVM的主流核函数
    if (model.type == "rbf"):
        for i in range(0, len(x)):
            # 注意python当中平方为:**
            result[i] = (x[i] - z[i]) ** 2
        finalResult = math.exp(-lamda * (sum(result) ** (1 / 2)))
    # 多项式函数，其中d也是需要进行调参设置的
    elif (model.type == "duoxiangshi"):
        d = 2  # 重点是阶数的选择
        r = 0  # d和r都是需要后续调整的参数
        for i in range(0, len(x)):
            # 注意python当中平方为:**
            result[i] = (x[i] * z[i])
        finalResult = (lamda * sum(result) + r) ** d
    else:
        print("error:输入的核函数类型不能识别")
    return finalResult


# 输入的x是自变量，应该也是一个向量才对
# 输入的x是完整的列表
# 返回的只也是一个向量
def gFunction(inputX, model):
    result = 0
    for j in range(0, model.trainCount):
        xj = model.trainData[j][0:11]
        tempHeResult = heFunction(inputX[0:11], xj, model)
        result = result + model.aerfa[j] * model.trainData[j][12] * tempHeResult

    result = result + model.yuzhiB
    return result


def whetherTerminate(model):
    result = True
    sum = 0
    for i in range(0, model.trainCount):
        sum = sum + model.aerfa[i] * model.trainData[i][12]

        # 判断0《=model.aerfai<=C
        if (not (model.aerfa[i] >= -model.e and model.aerfa[i] <= model.Cpara + model.e)):
            result = False
            break
        xi = model.trainData[i][0:11]

        if (model.aerfa[i] > 0 and model.aerfa[i] < model.Cpara and (not (
                model.trainData[i][12] * gFunction(xi, model) >= 1 - model.e or model.trainData[i][12] * gFunction(xi,
                                                                                                                   model) <= 1 - model.e))):
            result = False
            break
        if (model.aerfa[i] == 0 and (not model.trainData[i][12] * gFunction(xi, model) >= 1 - model.e)):
            result = False
            break
        if (model.aerfa[i] == model.Cpara and (not model.trainData[i][12] * gFunction(xi, model) <= 1 + model.e)):
            result = False
            break
    if (not (sum > -model.e and sum < -model.e)):
        result = False
    return result


def smo(model):
    # smoResult =False
    # 初始化E值
    for i in range(0, model.trainCount):
        model.EList[i] = - model.trainData[i][12]

    for h in range(0, model.maxEpcho):
        print("循环次数" + str(h))
        '外循环，搜寻第一个需要进行修改的α'
        for i in range(0, model.trainCount):
            xi = model.trainData[i][0:11]
            # 注意搜寻合适的阿尔法序列的时候
            # 检查样本是否满足KKT条件
            # 满足这个条件说明是支持向量，在边界上
            xiGFunctionValue = gFunction(xi, model)
            if (model.aerfa[i] > 0 and model.aerfa[i] < model.Cpara and (
            not model.trainData[i][12] * xiGFunctionValue == 1)):
                firstSelectedAIndex = i
            # 满足这个条件说明是正常分类，再边界内部
            elif (model.aerfa[i] == 0 and (not model.trainData[i][12] * xiGFunctionValue >= 1)):
                firstSelectedAIndex = i
            # 满足这个条件，说明再两个边界之间
            elif (model.aerfa[i] == model.Cpara and (not model.trainData[i][12] * xiGFunctionValue <= 1)):
                firstSelectedAIndex = i
            else:
                break
            '内循环：寻找第二个阿尔法的 我们觉得如果之前修改过的阿尔法，在这一步就不要再选择了'
            if (model.EList[firstSelectedAIndex] > 0):
                secondSelectedAIndex = model.EList.index(min(model.EList[firstSelectedAIndex:]))
            if (model.EList[firstSelectedAIndex] < 0):
                secondSelectedAIndex = model.EList.index(max(model.EList[firstSelectedAIndex:]))
            i = secondSelectedAIndex
            if (secondSelectedAIndex == -1):
                break
                print("smoError:寻找第二个修改的阿尔法值失败")

            # print( "找到的第二个需要修改的阿尔法的索引为：" + str(secondSelectedAIndex))
            '计算对应的L和H的值'
            L = max(0, model.aerfa[secondSelectedAIndex] - model.aerfa[firstSelectedAIndex])
            H = min(model.Cpara, model.Cpara + model.aerfa[secondSelectedAIndex] - model.aerfa[firstSelectedAIndex])
            # print("此时的L为：" + str(L))
            # print("此时的H为：" + str(H))

            model.aerfa2Old = model.aerfa[secondSelectedAIndex]
            model.aerfa1Old = model.aerfa[firstSelectedAIndex]
            # 修改第二个阿尔法的值
            '修改第二个阿尔法的数值'

            '计算没有修建的α的值'
            y1 = model.trainData[firstSelectedAIndex][12]
            y2 = model.trainData[secondSelectedAIndex][12]
            E1 = gFunction(model.trainData[firstSelectedAIndex][0:11], model) - model.trainData[firstSelectedAIndex][12]
            E2 = gFunction(model.trainData[secondSelectedAIndex][0:11], model) - model.trainData[secondSelectedAIndex][
                12]
            K11 = heFunction(model.trainData[firstSelectedAIndex][0:11], model.trainData[firstSelectedAIndex][0:11],
                             model)
            K22 = heFunction(model.trainData[secondSelectedAIndex][0:11], model.trainData[secondSelectedAIndex][0:11],
                             model)
            K12 = heFunction(model.trainData[firstSelectedAIndex][0:11], model.trainData[secondSelectedAIndex][0:11],
                             model)
            newA2Init = model.aerfa2Old + (y2 * (E1 - E2)) / (K11 + K22 - 2 * K12)

            '计算修建之后的阿尔法的值'
            if (newA2Init > H):
                model.aerfa[secondSelectedAIndex] = H
            elif (newA2Init >= L and newA2Init <= H):
                model.aerfa[secondSelectedAIndex] = newA2Init
            else:
                model.aerfa[secondSelectedAIndex] = L

            '计算α1新的值'
            # 根据a1+a2*y2*y1-a2'*y2*y1=a1'之间的线性关系，计算a1 new
            # temp:y1*y2
            temp = model.trainData[firstSelectedAIndex][12] * model.trainData[secondSelectedAIndex][12]
            # 修改第一个阿尔法的值
            model.aerfa[firstSelectedAIndex] = model.aerfa1Old + model.aerfa2Old * temp - model.aerfa[
                secondSelectedAIndex] * temp

            '更新阈值B'
            # 修改b
            # print(model.yuzhiB)
            b1New = model.yuzhiB - E1 - y1 * (model.aerfa[firstSelectedAIndex] - model.aerfa1Old) * K11 - y2 * (
                        model.aerfa[secondSelectedAIndex] - model.aerfa2Old) * K12
            b2New = model.yuzhiB - E2 - y1 * (model.aerfa[firstSelectedAIndex] - model.aerfa1Old) * K12 - y2 * (
                        model.aerfa[secondSelectedAIndex] - model.aerfa2Old) * K22
            if (0 < model.aerfa[secondSelectedAIndex] < model.Cpara):
                model.yuzhiB = b1New
            elif (0 < model.aerfa[firstSelectedAIndex] < model.Cpara):
                model.yuzhiB = b2New
            else:
                model.yuzhiB = (b1New + b2New) / 2
            # print("修改之前的阿尔法2的值为："+str(model.aerfa2Old)+"修改之后的阿尔法2的值"+str(model.aerfa[secondSelectedAIndex]))
            # print("修改之前的阿尔法1的值为：" + str(model.aerfa1Old) + "修改之后的阿尔法1的值" + str(model.aerfa[firstSelectedAIndex]))
            # 更新全部的E
            for j in range(0, model.trainCount):
                model.EList[j] = gFunction(model.trainData[j][0:11], model) - model.trainData[j][12]

            # 判断是否达到了终止条件
            smoResult = whetherTerminate(model)
            if (smoResult):
                break;

# if(not smoResult):
# print("没有在规定的次数当中找到最好的结果")
