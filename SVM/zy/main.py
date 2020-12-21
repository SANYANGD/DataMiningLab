import fileRead
import svm
import time
import numpy as np

# 初始化我们训练集和验证集
fileRead.initData()
result = [0] * 4
print("开始运行smo算法")
for i in range(0, 1):
    print(i)
    fileRead.initUsableData(1)

    trainData = fileRead.trainData

    # smoStartTime=time.perf_counter()
    model = svm.Model(trainData)
    svm.smo(model)
    # smoRunTime=time.perf_counter()-smoStartTime
    # print("smo算法运行时间%.2fs"  %smoRunTime)

    # print(yuzhiB,end="/")

    yuce = [0] * len(fileRead.testData)
    for i in range(0, len(fileRead.testData)):
        temp1 = 0
        for j in range(0, len(trainData)):
            temp1 = model.aerfa[j] * trainData[j][12] * svm.heFunction(fileRead.testData[i][0:11], trainData[j][0:11],
                                                                       model) + temp1
        yuce[i] = np.sign(temp1 + model.yuzhiB)
    Tp = [0] * 2
    Tn = [0] * 2
    Fp = [0] * 2
    Fn = [0] * 2

    for i in range(0, len(fileRead.testData)):
        if (fileRead.testData[i][12] == 1):
            if (yuce[i] == 1):
                Tp[1] = Tp[1] + 1
                Tn[0] = Tn[0] + 1
            if (yuce[i] == -1):
                Fn[1] = Fn[1] + 1
                Fp[0] = Fp[0] + 1
        if (fileRead.testData[i][12] == -1):
            if (yuce[i] == -1):
                Tp[0] = Tp[0] + 1
                Tn[1] = Tn[1] + 1
            if (yuce[i] == 1):
                Fn[0] = Fn[0] + 1
                Fp[1] = Fp[1] + 1

    accuracy = [0] * 2
    precision = [0] * 2
    recall = [0] * 2
    f1 = [0] * 2
    print(yuce)
    for k in range(0, len(fileRead.testData)):
        print(fileRead.testData[k][12], end=" ")
    print(model.aerfa)

    for i in range(0, 2):
        accuracy[i] = (Tp[i] + Tn[i]) / (Tp[i] + Tn[i] + Fp[i] + Fn[i])
        precision[i] = (Tp[i]) / (Tp[i] + Fp[i])
        recall[i] = Tp[i] / (Tp[i] + Fn[i])
        f1[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])

    result[0] = result[0] + sum(accuracy) / len(accuracy)
    result[1] = result[1] + sum(precision) / len(precision)
    result[2] = result[2] + sum(recall) / len(recall)
    result[3] = result[3] + sum(f1) / len(f1)

result[0] = result[0]
result[1] = result[1]
result[2] = result[2]
result[3] = result[3]

print("十折交叉验证的结果是：".center(50, '-'))
print("准确率为：%.2f" % result[0])
print("精确率为：%.2f" % result[1])
print("召回率为：%.2f" % result[2])
print("f1为：%.2f" % result[3])
